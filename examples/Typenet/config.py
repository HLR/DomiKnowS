device = 'cpu'
use_constraints = True
batch_size = 32

test_interval = 1

timing = False

freebase_only = True

if freebase_only:
	num_types = 1076
else:
	num_types = 1940

missing_types = ['__skiing__ski_lift', '__food__beer_containment', '__exhibitions__exhibition_producer', '__book__scholarly_financial_support']