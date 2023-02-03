import wget
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

CHECKPOINT_PATH = 'checkpoints'

checkpoint_urls = ["https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/baseline_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/baseline_500.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/digitlbl_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/digitlbl_500.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/explicit_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/explicit_500.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/primaldual_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/primaldual_500.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/samplingloss_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/samplingloss_500.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/semanticloss_10k.pth", "https://www.ocf.berkeley.edu/~alexwan/mnist-arithmetic-checkpoints/semanticloss_500.pth"]

if not os.path.isdir(CHECKPOINT_PATH):
	os.mkdir(CHECKPOINT_PATH)

for url in checkpoint_urls:
	print('downloading %s to %s' % (url, CHECKPOINT_PATH))
	wget.download(url, out=CHECKPOINT_PATH)
