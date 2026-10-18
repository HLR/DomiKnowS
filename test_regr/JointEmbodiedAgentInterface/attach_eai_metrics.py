import torch

p_src = torch.load('/workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260927_011044/joint_stage2_epoch_004.pt', map_location='cpu', weights_only=False)
p_dst = torch.load('/workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/joint_controller_warmup_384.pt', map_location='cpu', weights_only=False)
eai_metrics = p_src['metrics']['eai']
p_dst['metrics']['eai'] = eai_metrics
torch.save(p_dst, '/workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/joint_controller_warmup_384.pt')
print('Successfully attached eai metrics to joint_controller_warmup_384.pt:', eai_metrics)
