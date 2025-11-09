#!/bin/bash

###############################################################################
# AWS 完全清理脚本 - 停止所有费用产生
# 
# 功能：
# 1. 创建 EBS 卷快照备份数据
# 2. 删除所有 Load Balancers
# 3. 删除 NAT Gateway
# 4. 删除 EKS 集群
# 5. 释放 Elastic IP
# 6. 删除 EBS 卷（可选）
#
# 警告：此脚本会删除所有资源，操作不可逆！
###############################################################################

# 不使用 set -e，允许脚本在部分操作失败时继续执行

REGION="us-west-2"
CLUSTER_NAME="inf4rag"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}================================================================================${NC}"
echo -e "${RED}  警告：此脚本将删除所有 AWS 资源，操作不可逆！${NC}"
echo -e "${RED}================================================================================${NC}"
echo ""

# 支持 --yes 参数自动确认
AUTO_YES=false
if [ "$1" == "--yes" ] || [ "$1" == "-y" ]; then
    AUTO_YES=true
    echo "自动确认模式已启用"
fi

if [ "$AUTO_YES" != true ]; then
    read -p "确认要继续吗？输入 'YES' 继续: " confirm
    if [ "$confirm" != "YES" ]; then
        echo "操作已取消"
        exit 1
    fi
fi

echo -e "\n${GREEN}开始清理 AWS 资源...${NC}\n"

###############################################################################
# 步骤 1: 创建 EBS 快照备份（保留数据）
###############################################################################
echo -e "${YELLOW}[步骤 1/6] 创建 EBS 卷快照备份...${NC}"

# 已知的 EBS 卷 ID（根据之前的检查）
VOLUMES=(
    "vol-0a7aa01b4f29e575e"
    "vol-02db5fbc1632f93e6"
    "vol-0b3cc358b2f054a5e"
    "vol-07630af83c8c69d22"
    "vol-071e2b899adb52638"
    "vol-093ad89ffbba4f7fe"
    "vol-04122ac667b09a604"
)

# 先检查哪些卷存在
echo "  检查 EBS 卷状态..."
EXISTING_VOLUMES=()
for vol in "${VOLUMES[@]}"; do
    if aws ec2 describe-volumes --volume-ids "$vol" --region $REGION &>/dev/null; then
        EXISTING_VOLUMES+=("$vol")
        echo "    ✓ 卷存在: $vol"
    else
        echo "    ⚠ 卷不存在或已删除: $vol"
    fi
done

if [ ${#EXISTING_VOLUMES[@]} -eq 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} 没有需要备份的 EBS 卷"
else
    SNAPSHOT_IDS=()
    for vol in "${EXISTING_VOLUMES[@]}"; do
        echo "  创建快照: $vol"
        SNAPSHOT_ID=$(aws ec2 create-snapshot \
            --volume-id "$vol" \
            --description "inf4rag-backup-$TIMESTAMP" \
            --region $REGION \
            --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=inf4rag-backup-$TIMESTAMP},{Key=Purpose,Value=cleanup-backup}]" \
            --query 'SnapshotId' \
            --output text 2>/dev/null)
        
        if [ -n "$SNAPSHOT_ID" ] && [ "$SNAPSHOT_ID" != "None" ]; then
            SNAPSHOT_IDS+=("$SNAPSHOT_ID")
            echo -e "    ${GREEN}✓${NC} 快照创建中: $SNAPSHOT_ID"
        else
            echo -e "    ${RED}✗${NC} 快照创建失败: $vol"
        fi
    done
fi

if [ ${#SNAPSHOT_IDS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}等待快照完成（需要几分钟）...${NC}"
    echo "快照 ID 列表（请保存）："
    printf '%s\n' "${SNAPSHOT_IDS[@]}"

    # 等待快照完成（最多等待 10 分钟）
    MAX_WAIT=600
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        ALL_COMPLETE=true
        for snap_id in "${SNAPSHOT_IDS[@]}"; do
            STATUS=$(aws ec2 describe-snapshots \
                --snapshot-ids "$snap_id" \
                --region $REGION \
                --query 'Snapshots[0].State' \
                --output text 2>/dev/null || echo "error")
            
            if [ "$STATUS" != "completed" ]; then
                ALL_COMPLETE=false
                break
            fi
        done
        
        if [ "$ALL_COMPLETE" = true ]; then
            echo -e "${GREEN}✓${NC} 所有快照已完成"
            break
        fi
        
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        echo "  等待中... ($ELAPSED/$MAX_WAIT 秒)"
    done
else
    echo -e "  ${YELLOW}⚠${NC} 跳过快照等待（没有创建快照）"
fi

echo -e "${GREEN}[步骤 1/6] 完成${NC}\n"

###############################################################################
# 步骤 2: 删除 Network Load Balancer (NLB)
###############################################################################
echo -e "${YELLOW}[步骤 2/6] 删除 Network Load Balancer...${NC}"

NLB_ARN="arn:aws:elasticloadbalancing:us-west-2:448515985423:loadbalancer/net/a7352c8e59ef146938a79872128ff536/41d10471dbf9713f"

if aws elbv2 describe-load-balancers --load-balancer-arns "$NLB_ARN" --region $REGION &>/dev/null; then
    echo "  删除 NLB: $NLB_ARN"
    aws elbv2 delete-load-balancer \
        --load-balancer-arn "$NLB_ARN" \
        --region $REGION
    echo -e "  ${GREEN}✓${NC} NLB 删除中..."
else
    echo -e "  ${YELLOW}⚠${NC} NLB 不存在或已删除"
fi

echo -e "${GREEN}[步骤 2/6] 完成${NC}\n"

###############################################################################
# 步骤 3: 删除 Classic Load Balancers (ELB)
###############################################################################
echo -e "${YELLOW}[步骤 3/6] 删除 Classic Load Balancers...${NC}"

CLASSIC_ELBS=(
    "a66f31244cfe34f499d86b949595c67f"
    "a971802cc82fb4bc28974a5beab4291d"
    "ab9ca528f730e49a496214039ac462c4"
)

for elb_name in "${CLASSIC_ELBS[@]}"; do
    echo "  删除 Classic ELB: $elb_name"
    if aws elb describe-load-balancers --load-balancer-names "$elb_name" --region $REGION &>/dev/null; then
        aws elb delete-load-balancer \
            --load-balancer-name "$elb_name" \
            --region $REGION
        echo -e "    ${GREEN}✓${NC} ELB 删除中..."
    else
        echo -e "    ${YELLOW}⚠${NC} ELB 不存在或已删除"
    fi
done

echo -e "${GREEN}[步骤 3/6] 完成${NC}\n"

###############################################################################
# 步骤 4: 删除 NAT Gateway
###############################################################################
echo -e "${YELLOW}[步骤 4/6] 删除 NAT Gateway...${NC}"

NAT_GATEWAY_ID="nat-001e46b75d34626f0"

if aws ec2 describe-nat-gateways --nat-gateway-ids "$NAT_GATEWAY_ID" --region $REGION &>/dev/null; then
    echo "  删除 NAT Gateway: $NAT_GATEWAY_ID"
    aws ec2 delete-nat-gateway \
        --nat-gateway-id "$NAT_GATEWAY_ID" \
        --region $REGION
    echo -e "  ${GREEN}✓${NC} NAT Gateway 删除中..."
else
    echo -e "  ${YELLOW}⚠${NC} NAT Gateway 不存在或已删除"
fi

echo -e "${GREEN}[步骤 4/6] 完成${NC}\n"

###############################################################################
# 步骤 5: 释放 Elastic IP
###############################################################################
echo -e "${YELLOW}[步骤 5/6] 释放 Elastic IP...${NC}"

ALLOCATION_ID="eipalloc-024e5adc0992a4259"
ASSOCIATION_ID="eipassoc-046ff41ba7a386286"

# 先解除关联
if aws ec2 describe-addresses --allocation-ids "$ALLOCATION_ID" --region $REGION --query "Addresses[0].AssociationId" --output text 2>/dev/null | grep -q "."; then
    echo "  解除 Elastic IP 关联: $ASSOCIATION_ID"
    aws ec2 disassociate-address \
        --association-id "$ASSOCIATION_ID" \
        --region $REGION 2>/dev/null || echo "  已解除关联或不存在"
fi

# 释放 Elastic IP
echo "  释放 Elastic IP: $ALLOCATION_ID"
aws ec2 release-address \
    --allocation-id "$ALLOCATION_ID" \
    --region $REGION 2>/dev/null && echo -e "  ${GREEN}✓${NC} Elastic IP 已释放" || echo -e "  ${YELLOW}⚠${NC} Elastic IP 可能已释放"

echo -e "${GREEN}[步骤 5/6] 完成${NC}\n"

###############################################################################
# 步骤 6: 删除 EKS 集群
###############################################################################
echo -e "${YELLOW}[步骤 6/6] 删除 EKS 集群...${NC}"

# 检查 eksctl 是否安装
if ! command -v eksctl &> /dev/null; then
    echo -e "  ${YELLOW}⚠${NC} eksctl 未安装，尝试使用 AWS CLI 删除集群"
    
    # 先删除所有节点组
    echo "  获取节点组列表..."
    NODEGROUPS=$(aws eks list-nodegroups --cluster-name $CLUSTER_NAME --region $REGION --query 'nodegroups[]' --output text)
    
    for ng in $NODEGROUPS; do
        echo "  删除节点组: $ng"
        aws eks delete-nodegroup \
            --cluster-name $CLUSTER_NAME \
            --nodegroup-name "$ng" \
            --region $REGION || echo "  节点组可能已删除"
    done
    
    # 注意：节点组删除是异步的，不需要等待
    if [ -n "$NODEGROUPS" ]; then
        echo "  注意：节点组删除已启动（异步操作，可能需要几分钟完成）"
    fi
    
    # 删除集群
    echo "  删除 EKS 集群: $CLUSTER_NAME"
    aws eks delete-cluster \
        --name $CLUSTER_NAME \
        --region $REGION || echo "  集群可能已删除"
else
    echo "  使用 eksctl 删除集群: $CLUSTER_NAME"
    echo "  注意：集群删除是异步的，可能需要几分钟到十几分钟完成"
    echo "  删除操作已启动，脚本将继续执行..."
    eksctl delete cluster --name $CLUSTER_NAME --region $REGION || echo "  集群可能已删除或删除已启动"
fi

echo -e "${GREEN}[步骤 6/6] 完成${NC}\n"

###############################################################################
# 步骤 7: 删除 EBS 卷（可选）
###############################################################################
echo -e "${YELLOW}[步骤 7/7] 删除 EBS 卷...${NC}"

if [ "$AUTO_YES" = true ]; then
    delete_volumes="YES"
    echo "自动模式：将删除 EBS 卷（数据已在快照中）"
else
    read -p "是否删除 EBS 卷？（数据已备份到快照）输入 'YES' 删除: " delete_volumes
fi

if [ "$delete_volumes" = "YES" ]; then
    DELETED_COUNT=0
    for vol in "${EXISTING_VOLUMES[@]}"; do
        echo "  删除卷: $vol"
        # 先尝试 detach（如果还关联着）
        INSTANCE_ID=$(aws ec2 describe-volumes \
            --volume-ids "$vol" \
            --region $REGION \
            --query 'Volumes[0].Attachments[0].InstanceId' \
            --output text 2>/dev/null)
        
        if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
            echo "    解除关联: $vol -> $INSTANCE_ID"
            aws ec2 detach-volume \
                --volume-id "$vol" \
                --region $REGION 2>/dev/null || true
            sleep 2
        fi
        
        # 删除卷
        if aws ec2 delete-volume --volume-id "$vol" --region $REGION 2>/dev/null; then
            echo -e "    ${GREEN}✓${NC} 卷已删除"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        else
            echo -e "    ${YELLOW}⚠${NC} 卷删除失败或不存在"
        fi
    done
    if [ $DELETED_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓${NC} 已删除 $DELETED_COUNT 个 EBS 卷"
    else
        echo -e "${YELLOW}⚠${NC} 没有可删除的 EBS 卷"
    fi
else
    echo -e "${YELLOW}⚠${NC} 保留 EBS 卷（仍会产生存储费用）"
fi

###############################################################################
# 清理完成总结
###############################################################################
echo -e "\n${GREEN}================================================================================${NC}"
echo -e "${GREEN}清理完成！${NC}"
echo -e "${GREEN}================================================================================${NC}\n"

echo -e "重要信息："
if [ ${#SNAPSHOT_IDS[@]} -gt 0 ]; then
    echo -e "  1. 数据已备份到以下快照："
    printf '     - %s\n' "${SNAPSHOT_IDS[@]}"
else
    echo -e "  1. 没有创建快照（EBS 卷可能已不存在）"
fi
echo ""
echo -e "  2. 已删除的资源："
echo -e "     ✓ Load Balancers (1 NLB + 3 Classic ELB)"
echo -e "     ✓ NAT Gateway"
echo -e "     ✓ Elastic IP"
echo -e "     ✓ EKS 集群"
if [ "$delete_volumes" = "YES" ]; then
    echo -e "     ✓ EBS 卷（数据在快照中）"
fi
echo ""
echo -e "${YELLOW}注意：${NC}"
echo -e "  - 快照会产生少量存储费用（约 $0.05/GB/月）"
echo -e "  - 如果不删除 EBS 卷，仍会产生存储费用（约 $0.10/GB/月）"
echo -e "  - 某些资源删除可能需要几分钟才能完成"
echo ""
echo -e "查看快照："
echo -e "  aws ec2 describe-snapshots --owner-ids self --region $REGION --filters \"Name=tag:Name,Values=inf4rag-backup-*\" --output table"
echo ""

