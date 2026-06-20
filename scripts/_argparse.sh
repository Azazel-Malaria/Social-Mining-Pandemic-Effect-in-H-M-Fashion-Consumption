#!/usr/bin/env bash
# Shared lightweight parser: scripts can be run as
# bash script.sh --data_root ./data --cuda_id 0 --output_root ./output
# It supports both --key value and --key=value. Bare flags become 1.
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --*=*)
      name="${key%%=*}"
      value="${key#*=}"
      name="${name#--}"
      export "$name"="$value"
      shift
      ;;
    --*)
      name="${key#--}"
      if [[ $# -ge 2 && "${2:-}" != --* ]]; then
        value="$2"
        shift 2
      else
        value="1"
        shift 1
      fi
      export "$name"="$value"
      ;;
    *)
      shift
      ;;
  esac
done
