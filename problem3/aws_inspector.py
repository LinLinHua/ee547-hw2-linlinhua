#!/usr/bin/env python3
"""
Problem 3 – AWS Resource Inspector

What this script does (read-only):
  • Auth via AWS CLI creds or env vars and verify with STS GetCallerIdentity
  • Collect IAM users, EC2 instances, S3 buckets, and Security Groups
  • Output in the required JSON shape (default) or a pretty table
  • Graceful error handling: auth, permissions, timeouts, invalid region, empty resources

Allowed packages: boto3 + Python standard library only.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Any

import boto3
import botocore


# ----------------------------- small utilities -----------------------------

def iso(dt) -> str | None:
    """Return ISO-8601 string or None."""
    return dt.isoformat().replace("+00:00", "Z") if dt else None

def iso_now_z() -> str:
    """Current UTC timestamp in ISO-8601 with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def warn(msg: str) -> None:
    """Print a WARNING to stderr."""
    print(f"[WARNING] {msg}", file=sys.stderr)

def error(msg: str) -> None:
    """Print an ERROR to stderr."""
    print(f"[ERROR] {msg}", file=sys.stderr)

def retry(call: Callable[[], Any], tries: int = 3, delay: float = 0.6):
    """
    Tiny retry wrapper for flaky network calls (EndpointConnectionError).
    Raises on final failure.
    """
    for i in range(tries):
        try:
            return call()
        except botocore.exceptions.EndpointConnectionError as e:
            if i == tries - 1:
                raise
            time.sleep(delay)


# ----------------------------- Part A – Auth -------------------------------

def make_session(region: str | None) -> boto3.Session:
    """
    Create a boto3 Session. Verify credentials via STS GetCallerIdentity.
    Exit with code 1 on auth errors (as required by spec).
    """
    try:
        session = boto3.Session(region_name=region)
        sts = session.client("sts")
        ident = sts.get_caller_identity()  # raises if creds invalid
        print(f"[auth] OK – Account={ident.get('Account')} Arn={ident.get('Arn')}", file=sys.stderr)
        return session
    except botocore.exceptions.NoCredentialsError:
        error("No AWS credentials found. Run `aws configure` or set env vars.")
        sys.exit(1)
    except botocore.exceptions.ClientError as e:
        error(f"STS/GetCallerIdentity failed: {e}")
        sys.exit(1)


# ----------------------- Part D – Resource collectors ----------------------

# IAM Users -----------------------------------------------------------------

def collect_iam(session: boto3.Session) -> dict:
    """
    Output:
      {"users":[
         {"username","user_id","arn","create_date","last_activity",
          "attached_policies":[{"policy_name","policy_arn"}, ...]
         }, ...
      ]}
    Required perms: iam:ListUsers, iam:GetUser, iam:ListAttachedUserPolicies
    """
    iam = session.client("iam")
    users_out = []
    try:
        paginator = iam.get_paginator("list_users")
        for page in retry(lambda: paginator.paginate()):
            for u in page.get("Users", []) or []:
                uname = u["UserName"]
                item = {
                    "username": uname,
                    "user_id": u.get("UserId"),
                    "arn": u.get("Arn"),
                    "create_date": iso(u.get("CreateDate")),
                    "last_activity": None,
                    "attached_policies": [],
                }
                # PasswordLastUsed is returned by GetUser if available
                try:
                    gu = iam.get_user(UserName=uname).get("User", {})
                    item["last_activity"] = iso(gu.get("PasswordLastUsed"))
                except botocore.exceptions.ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    item["last_activity"] = None
                    warn(f"GetUser failed for '{uname}': {code}")

                try:
                    pols = iam.list_attached_user_policies(UserName=uname).get("AttachedPolicies", [])
                    item["attached_policies"] = [
                        {"policy_name": p.get("PolicyName"), "policy_arn": p.get("PolicyArn")} for p in pols
                    ]
                except botocore.exceptions.ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    warn(f"ListAttachedUserPolicies failed for '{uname}': {code}")

                users_out.append(item)
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        warn(f"Access denied for IAM operations – skipping user enumeration ({code})")
        return {"users": []}
    except botocore.exceptions.EndpointConnectionError:
        warn("Network error while listing IAM users – skipping")
        return {"users": []}

    if not users_out:
        warn("No IAM users found")
    return {"users": users_out}


# EC2 Instances (+ SG summary fields) --------------------------------------

def collect_ec2(session: boto3.Session) -> dict:
    """
    Output:
      {
        "instances":[{ instance_id, instance_type, state, public_ip, private_ip,
                       availability_zone, launch_time, ami_id, ami_name,
                       security_groups:[sg-...], tags:{...} }, ...],
        "security_groups":[{group_id, group_name, vpc_id, description,
                            inbound_rules:int, outbound_rules:int}, ...]
      }
    Required perms: ec2:DescribeInstances, ec2:DescribeImages, ec2:DescribeSecurityGroups
    """
    ec2 = session.client("ec2")
    instances_raw = []
    try:
        paginator = ec2.get_paginator("describe_instances")
        for page in retry(lambda: paginator.paginate()):
            for res in page.get("Reservations", []) or []:
                instances_raw.extend(res.get("Instances", []) or [])
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        warn(f"Access denied for EC2 operations – skipping instances ({code})")
        return {"instances": [], "security_groups": []}
    except botocore.exceptions.EndpointConnectionError:
        warn("Network error while describing EC2 instances – skipping")
        return {"instances": [], "security_groups": []}

    # collect AMI names in batch
    ami_ids = sorted({i.get("ImageId") for i in instances_raw if i.get("ImageId")})
    ami_name_by_id = {}
    for i in range(0, len(ami_ids), 100):
        batch = ami_ids[i:i + 100]
        if not batch:
            continue
        try:
            imgs = ec2.describe_images(ImageIds=batch).get("Images", [])
            for img in imgs:
                ami_name_by_id[img.get("ImageId")] = img.get("Name")
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            warn(f"DescribeImages failed for {len(batch)} AMIs: {code}")

    instances = []
    for inst in instances_raw:
        sg_ids = [g.get("GroupId") for g in inst.get("SecurityGroups", []) or [] if g.get("GroupId")]
        tags = {t["Key"]: t.get("Value") for t in inst.get("Tags", [])} if inst.get("Tags") else {}
        ami_id = inst.get("ImageId")
        instances.append({
            "instance_id": inst.get("InstanceId"),
            "instance_type": inst.get("InstanceType"),
            "state": (inst.get("State") or {}).get("Name"),
            "public_ip": inst.get("PublicIpAddress"),
            "private_ip": inst.get("PrivateIpAddress"),
            "availability_zone": (inst.get("Placement") or {}).get("AvailabilityZone"),
            "launch_time": iso(inst.get("LaunchTime")),
            "ami_id": ami_id,
            "ami_name": ami_name_by_id.get(ami_id),
            "security_groups": sg_ids,
            "tags": tags,
        })

    # SG summary (counts of rules)
    sgs = []
    try:
        pg = ec2.get_paginator("describe_security_groups")
        for page in retry(lambda: pg.paginate()):
            for sg in page.get("SecurityGroups", []) or []:
                sgs.append({
                    "group_id": sg.get("GroupId"),
                    "group_name": sg.get("GroupName"),
                    "vpc_id": sg.get("VpcId"),
                    "description": sg.get("Description"),
                    "inbound_rules": len(sg.get("IpPermissions", []) or []),
                    "outbound_rules": len(sg.get("IpPermissionsEgress", []) or []),
                })
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        warn(f"DescribeSecurityGroups denied – SG summary omitted ({code})")

    if not instances:
        warn(f"No EC2 instances found in {session.region_name or 'configured region'}")
    return {"instances": instances, "security_groups": sgs}


# Security Groups (full rule expansion needed by spec) ----------------------

def _fmt_protocol(ip_proto: str | None) -> str:
    return "all" if ip_proto in (None, "-1") else ip_proto.lower()

def _fmt_port_range(perm: dict) -> str:
    fp = perm.get("FromPort")
    tp = perm.get("ToPort")
    if fp is None or tp is None:
        return "all"
    return f"{fp}-{tp}"

def _expand_peers(perm: dict) -> list[str]:
    peers: list[str] = []
    for r in perm.get("IpRanges", []) or []:
        if r.get("CidrIp"):
            peers.append(r["CidrIp"])
    for r in perm.get("Ipv6Ranges", []) or []:
        if r.get("CidrIpv6"):
            peers.append(r["CidrIpv6"])
    for r in perm.get("UserIdGroupPairs", []) or []:
        if r.get("GroupId"):
            peers.append(r["GroupId"])
    for r in perm.get("PrefixListIds", []) or []:
        if r.get("PrefixListId"):
            peers.append(r["PrefixListId"])
    return peers or ["all"]

def collect_security_groups(session: boto3.Session) -> dict:
    """
    Output: {"security_groups":[
      {"group_id","group_name","description","vpc_id",
       "inbound_rules":[{"protocol","port_range","source"}],
       "outbound_rules":[{"protocol","port_range","destination"}]
      }, ...]}
    """
    ec2 = session.client("ec2")
    sgs_out = []
    try:
        paginator = ec2.get_paginator("describe_security_groups")
        for page in retry(lambda: paginator.paginate()):
            for sg in page.get("SecurityGroups", []) or []:
                inbound = []
                for p in sg.get("IpPermissions", []) or []:
                    proto = _fmt_protocol(p.get("IpProtocol"))
                    pr = _fmt_port_range(p)
                    for src in _expand_peers(p):
                        inbound.append({"protocol": proto, "port_range": pr, "source": src})

                outbound = []
                for p in sg.get("IpPermissionsEgress", []) or []:
                    proto = _fmt_protocol(p.get("IpProtocol"))
                    pr = _fmt_port_range(p)
                    for dst in _expand_peers(p):
                        outbound.append({"protocol": proto, "port_range": pr, "destination": dst})

                sgs_out.append({
                    "group_id": sg.get("GroupId"),
                    "group_name": sg.get("GroupName"),
                    "description": sg.get("Description"),
                    "vpc_id": sg.get("VpcId"),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound,
                })
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        warn(f"DescribeSecurityGroups denied – skipping detailed SG rules ({code})")
        return {"security_groups": []}
    except botocore.exceptions.EndpointConnectionError:
        warn("Network error while listing Security Groups – skipping")
        return {"security_groups": []}

    return {"security_groups": sgs_out}


# S3 Buckets ----------------------------------------------------------------

def collect_s3(session: boto3.Session) -> dict:
    """
    Output:
      {"buckets":[
        {"bucket_name","creation_date","region","object_count","size_bytes"}, ...
      ]}
    We approximate object_count/size_bytes by paginating ListObjectsV2.
    Required perms: s3:ListAllMyBuckets, s3:GetBucketLocation, s3:ListBucket
    """
    s3 = session.client("s3")
    out = []
    try:
        resp = retry(lambda: s3.list_buckets())
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        warn(f"Access denied for S3 list – skipping buckets ({code})")
        return {"buckets": []}
    except botocore.exceptions.EndpointConnectionError:
        warn("Network error while listing buckets – skipping")
        return {"buckets": []}

    for b in resp.get("Buckets", []) or []:
        name = b["Name"]
        created = iso(b.get("CreationDate"))

        # bucket region
        try:
            loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
            region = loc or "us-east-1"
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            region = f"error:{code}"

        # approximate size & count
        obj_count: int | str = 0
        size_bytes: int | str = 0
        try:
            s3_regional = session.client("s3", region_name=None if str(region).startswith("error:") else region)
            paginator = s3_regional.get_paginator("list_objects_v2")
            for page in retry(lambda: paginator.paginate(Bucket=name)):
                for obj in page.get("Contents", []) or []:
                    obj_count += 1
                    size_bytes += int(obj.get("Size", 0))
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            warn(f"Failed to list objects for bucket '{name}': {code}")
            obj_count = f"error:{code}"
            size_bytes = f"error:{code}"
        except botocore.exceptions.EndpointConnectionError:
            warn(f"Network error listing objects for bucket '{name}'")
            obj_count = "error:network"
            size_bytes = "error:network"

        out.append({
            "bucket_name": name,
            "creation_date": created,
            "region": region,
            "object_count": obj_count,
            "size_bytes": size_bytes,
        })

    if not out:
        warn("No S3 buckets found")
    return {"buckets": out}


# ----------------------- Part E – Output formatting ------------------------

def get_account_info(session: boto3.Session, region_hint: str | None) -> dict:
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    return {
        "account_id": ident.get("Account"),
        "user_arn": ident.get("Arn"),
        "region": region_hint or session.region_name,
        "scan_timestamp": iso_now_z(),
    }

def to_json_blob(account_info: dict, iam_d: dict, ec2_d: dict, s3_d: dict, sg_d: dict) -> dict:
    users = iam_d.get("users", [])
    instances = ec2_d.get("instances", [])
    buckets = s3_d.get("buckets", [])
    sgs = sg_d.get("security_groups", [])

    running = sum(1 for i in instances if i.get("state") == "running")

    return {
        "account_info": account_info,
        "resources": {
            "iam_users": users,
            "ec2_instances": instances,
            "s3_buckets": buckets,
            "security_groups": sgs,
        },
        "summary": {
            "total_users": len(users),
            "running_instances": running,
            "total_buckets": len(buckets),
            "security_groups": len(sgs),
        },
    }

def print_table(blob: dict, fp) -> None:
    """Pretty text table resembling the spec screenshot."""
    acct = blob["account_info"]
    res  = blob["resources"]
    sumy = blob["summary"]

    print("Table Format", file=fp)
    print(f"\nAWS Account: {acct.get('account_id')} ({acct.get('region')})", file=fp)
    print(f"Scan Time: {acct.get('scan_timestamp')} UTC", file=fp)

    users = res.get("iam_users", []) or []
    print(f"\nIAM USERS ({sumy.get('total_users',0)} total)", file=fp)
    print(f"{'Username':<30} {'Create Date':<20} {'Last Activity':<20} {'Policies':<8}", file=fp)
    for u in users:
        print(f"{u.get('username',''):<30} "
              f"{(u.get('create_date') or '-'):<20} "
              f"{(u.get('last_activity') or '-'):<20} "
              f"{len(u.get('attached_policies', [])):<8}", file=fp)
    if not users:
        print("(none)", file=fp)

    insts = res.get("ec2_instances", []) or []
    running = sum(1 for i in insts if i.get("state") == "running")
    stopped = sum(1 for i in insts if i.get("state") == "stopped")
    print(f"\nEC2 INSTANCES ({running} running, {stopped} stopped)", file=fp)
    print(f"{'Instance ID':<20} {'Type':<10} {'State':<10} {'Public IP':<15} {'Launch Time':<19}", file=fp)
    for i in insts:
        print(f"{i.get('instance_id',''):<20} {i.get('instance_type',''):<10} {i.get('state',''):<10} "
              f"{(i.get('public_ip') or '–'):<15} {(i.get('launch_time') or '-'):<19}", file=fp)
    if not insts:
        print("(none)", file=fp)

    buckets = res.get("s3_buckets", []) or []
    print(f"\nS3 BUCKETS ({len(buckets)} total)", file=fp)
    print(f"{'Bucket Name':<40} {'Region':<12} {'Created':<20} {'Objects':>8} {'Size (MB)':>10}", file=fp)
    for b in buckets:
        sz = b.get("size_bytes")
        size_mb = f"{(sz/1_000_000):.1f}" if isinstance(sz, (int, float)) else str(sz).replace("error:", "~")
        oc = b.get("object_count")
        objc = f"{oc}" if isinstance(oc, (int, float)) else str(oc).replace("error:", "~")
        print(f"{b.get('bucket_name',''):<40} {b.get('region',''):<12} "
              f"{(b.get('creation_date') or '-'):<20} {objc:>8} {size_mb:>10}", file=fp)
    if not buckets:
        print("(none)", file=fp)

    sgs = res.get("security_groups", []) or []
    print(f"\nSECURITY GROUPS ({len(sgs)} total)", file=fp)
    print(f"{'Group ID':<18} {'Name':<20} {'VPC ID':<15} {'Inbound Rules':>13}", file=fp)
    for g in sgs:
        print(f"{g.get('group_id',''):<18} {g.get('group_name',''):<20} {g.get('vpc_id',''):<15} "
              f"{len(g.get('inbound_rules', [])):>13}", file=fp)
    if not sgs:
        print("(none)", file=fp)


# ------------------------------- CLI / main --------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Read-only inspector for IAM users, EC2, S3, and Security Groups."
    )
    p.add_argument("--region", type=str, default=None,
                   help="AWS region to inspect (default: from credentials/config)")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path (default: print to stdout)")
    p.add_argument("--format", type=str, default="json", choices=["json", "table"],
                   help="Output format (default: json)")
    return p.parse_args()

def main():
    args = parse_args()

    # Create session and verify auth (exit on failure)
    session = make_session(args.region)

    # Collect resources (each collector handles its own warnings)
    iam_d = collect_iam(session)
    ec2_d = collect_ec2(session)
    s3_d  = collect_s3(session)
    sg_d  = collect_security_groups(session)

    # Build final JSON blob (account_info/resources/summary)
    acct = get_account_info(session, region_hint=args.region)
    blob = to_json_blob(acct, iam_d, ec2_d, s3_d, sg_d)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(blob, f, ensure_ascii=False, indent=2)
            else:
                print_table(blob, f)
        print(f"[saved] {os.path.abspath(args.output)}")
    else:
        if args.format == "json":
            json.dump(blob, sys.stdout, ensure_ascii=False, indent=2)
            print()
        else:
            print_table(blob, sys.stdout)

if __name__ == "__main__":
    main()
