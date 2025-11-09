# AWS 资源清理脚本使用说明

## 概述

`aws-cleanup-all.sh` 脚本用于完全清理 AWS 资源，停止所有费用产生。

## 功能

脚本会按顺序执行以下操作：

1. **创建 EBS 快照备份** - 在删除前备份所有数据
2. **删除 Network Load Balancer (NLB)** - 节省 ~$16/月
3. **删除 Classic Load Balancers (ELB)** - 节省 ~$54/月
4. **删除 NAT Gateway** - 节省 ~$32/月
5. **释放 Elastic IP** - 节省 ~$4/月
6. **删除 EKS 集群** - 节省 ~$72/月
7. **删除 EBS 卷（可选）** - 节省 ~$56/月

**总计可节省：~$234/月**

## 使用方法

### 前置要求

1. 已安装并配置 AWS CLI
   ```bash
   aws --version
   aws configure list
   ```

2. （可选）已安装 eksctl（用于删除 EKS 集群）
   ```bash
   eksctl version
   ```

### 运行脚本

```bash
cd /Users/weianqi/Projects/inf4RAG
./aws-cleanup-all.sh
```

### 操作流程

1. **确认操作** - 脚本会要求输入 `YES` 确认
2. **创建快照** - 自动为所有 EBS 卷创建快照（需要几分钟）
3. **删除资源** - 按顺序删除所有资源
4. **选择是否删除 EBS 卷** - 脚本会询问是否删除 EBS 卷（数据已在快照中）

## 重要提示

⚠️ **警告：此操作不可逆！**

- 脚本会删除所有资源，请确保已备份重要数据
- 快照会自动创建，但如果选择删除 EBS 卷，原始数据将丢失
- 某些资源删除可能需要几分钟到几十分钟才能完成

## 费用说明

### 完全清理后
- 所有按小时计费的资源已删除
- 快照存储费用：约 $0.05/GB/月（7个快照 × 80GB = 560GB ≈ $28/月）

### 如果保留 EBS 卷
- EBS 存储费用：约 $0.10/GB/月（7个卷 × 80GB = 560GB ≈ $56/月）
- 但数据立即可用，无需恢复

## 恢复数据

如果将来需要恢复数据：

```bash
# 1. 查看快照
aws ec2 describe-snapshots \
  --owner-ids self \
  --region us-west-2 \
  --filters "Name=tag:Name,Values=inf4rag-backup-*" \
  --output table

# 2. 从快照创建新卷
aws ec2 create-volume \
  --snapshot-id <snapshot-id> \
  --availability-zone us-west-2a \
  --region us-west-2 \
  --volume-type gp3

# 3. 附加到实例
aws ec2 attach-volume \
  --volume-id <volume-id> \
  --instance-id <instance-id> \
  --device /dev/xvdf \
  --region us-west-2
```

## 验证清理结果

清理完成后，检查是否还有资源：

```bash
# 检查 Load Balancers
aws elbv2 describe-load-balancers --region us-west-2
aws elb describe-load-balancers --region us-west-2

# 检查 NAT Gateway
aws ec2 describe-nat-gateways --region us-west-2

# 检查 EKS 集群
aws eks list-clusters --region us-west-2

# 检查 Elastic IP
aws ec2 describe-addresses --region us-west-2

# 检查 EBS 卷
aws ec2 describe-volumes --region us-west-2
```

## 故障排除

### 如果脚本执行失败

1. 检查 AWS 凭证：
   ```bash
   aws sts get-caller-identity
   ```

2. 检查权限：
   确保 IAM 用户有删除这些资源的权限

3. 手动删除剩余资源：
   查看脚本输出，找到失败的步骤，手动执行对应的 AWS CLI 命令

### 如果某些资源删除失败

某些资源可能有依赖关系，需要先删除依赖的资源。脚本会尝试自动处理，但如果失败，可以：

1. 查看 AWS 控制台的错误信息
2. 手动删除依赖的资源
3. 重新运行脚本或手动删除剩余资源

## 联系方式

如有问题，请检查：
- AWS 控制台：https://console.aws.amazon.com
- AWS 文档：https://docs.aws.amazon.com

