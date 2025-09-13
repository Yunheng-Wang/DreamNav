#!/usr/bin/env bash
set -euo pipefail

# ===== 参数接收 =====
ROBOT_USER="${1:-agilex}"                       # 默认 agilex
ROBOT_IP="${2:-10.12.143.140}"                  # 默认 10.12.143.140
LOCAL_SAVE_DIR="${3:-/home/dreams/tmp}"         # 默认 /home/dreams/tmp
ROS_DISTRO="melodic"                            # 固定
SAVER_NODES=("/rgb_saver" "/depth_saver")       # 不改名
# ====================

mkdir -p "${LOCAL_SAVE_DIR}"

# 复用 SSH 连接（显著减少每次握手时延）
CTRL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="${CTRL_PATH}" -o ControlPersist=60 -o StrictHostKeyChecking=accept-new)

# 预热控制连接（后台常驻 60s）
ssh -f -N "${SSH_OPTS[@]}" "${ROBOT_USER}@${ROBOT_IP}" || true

fetch_one() {
  local saver_node="$1"

  # 在小车上触发保存并回传“最新文件完整路径”
  local remote_last_file
  remote_last_file=$(
    ssh "${SSH_OPTS[@]}" "${ROBOT_USER}@${ROBOT_IP}" "
      set -e
      source /opt/ros/${ROS_DISTRO}/setup.bash
      rosservice list | grep -q \"${saver_node}/save\"
      rosservice call ${saver_node}/save >/dev/null
      FMT=\$(rosparam get ${saver_node#/}/filename_format 2>/dev/null || rosparam get image_saver/filename_format)
      PAT=\$(echo \"\${FMT}\" | sed 's/%0[0-9]d/*/g; s/%d/*/g')
      case \"\${PAT}\" in
        /*) SEARCH=\"\${PAT}\" ;;
        *)  SEARCH=\"/tmp/\${PAT}\" ;;
      esac
      ls -t \${SEARCH} 2>/dev/null | head -1
    "
  )

  if [[ -z "${remote_last_file}" ]]; then
    echo "❌ ${saver_node}: 没找到刚保存的文件" >&2
    return 1
  fi
  echo "✅ ${saver_node}: 小车最新文件：${remote_last_file}"

  # 按“原文件名”拷回服务器（不改名，保留时间戳/权限）
  local local_file="${LOCAL_SAVE_DIR}/$(basename "${remote_last_file}")"
  scp -p -o ControlPath="${CTRL_PATH}" "${ROBOT_USER}@${ROBOT_IP}:${remote_last_file}" "${local_file}"
  echo "✅ ${saver_node}: 已保存到服务器：${local_file}"
}

# 并行抓取 RGB 与 Depth
pids=()
for node in "${SAVER_NODES[@]}"; do
  fetch_one "$node" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
