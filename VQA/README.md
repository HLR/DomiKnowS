# Setup

## Download Data
1) Download data (~230mb) @ [Google Drive](https://drive.google.com/drive/folders/16JUM9iH-gCCs-uNjnvObApGfjEFZzIim?usp=sharing).
2) Copy `data/` folder to this examples folder

## Run

`python main.py`

# Results
This task involves the classification of objects in a scene. Objects are classified with respec to a hierarchy of labels (see: `hierarchy.json`). The hierarchy has four levels (i.e., each object has a maximum of four labels).

## Overall metrics
The following table shows classification metrics for each of the four levels with and without `GBI`:

|         |                  |         | Macro avg |         |         | Weighted Avg |         |         |       |
| ------- | ---------------- | ------- | --------- | ------- | ------- | ------------ | ------- | ------- | ----- |
| Level   | Method           | Acc     | P         | R       | F1      | P            | R       | F1      | n     |
| Level 1 | level1           | 58.76%  | 37.98%    | 32.02%  | 33.04%  | 56.32%       | 58.76%  | 56.60%  | 27155 |
| Level 1 | level1           | 58.57%  | 37.63%    | 30.38%  | 31.80%  | 56.08%       | 58.57%  | 56.21%  | 27155 |
|         | w/ GBI - w/o GBI | \-0.18% | \-0.36%   | \-1.64% | \-1.24% | \-0.24%      | \-0.18% | \-0.39% |       |
|         |                  |         |           |         |         |              |         |         |       |
| Level 2 | level2           | 65.49%  | 49.56%    | 35.65%  | 39.16%  | 63.91%       | 65.49%  | 63.67%  | 27155 |
| Level 2 | level2           | 65.41%  | 52.30%    | 33.15%  | 37.65%  | 64.04%       | 65.41%  | 63.42%  | 27155 |
|         | w/ GBI - w/o GBI | \-0.07% | 2.74%     | \-2.50% | \-1.51% | 0.13%        | \-0.07% | \-0.25% |       |
|         |                  |         |           |         |         |              |         |         |       |
| Level 3 | level3           | 81.18%  | 32.55%    | 46.38%  | 36.70%  | 85.82%       | 81.18%  | 82.74%  | 27155 |
| Level 3 | level3           | 85.87%  | 42.20%    | 37.48%  | 36.68%  | 85.83%       | 85.87%  | 85.69%  | 27155 |
|         | w/ GBI - w/o GBI | 4.69%   | 9.65%     | \-8.89% | \-0.01% | 0.02%        | 4.69%   | 2.95%   |       |
|         |                  |         |           |         |         |              |         |         |       |
| Level 4 | level4           | 95.65%  | 19.05%    | 73.45%  | 23.70%  | 99.48%       | 95.65%  | 97.42%  | 27155 |
| Level 5 | level4           | 97.71%  | 26.20%    | 64.49%  | 32.46%  | 99.49%       | 97.71%  | 98.53%  | 27155 |
|         | w/ GBI - w/o GBI | 2.06%   | 7.15%     | \-8.95% | 8.76%   | 0.00%        | 2.06%   | 1.11%   |

## Satisfied constraints
The following table shows classification metrics for each of the four levels with and without `GBI`, but limited to samples where the model initially does not satisfy constraints and `GBI` satisfies constraints:

|         |                  |         | Macro avg |          |         | Weighted Avg |         |         |      |
| ------- | ---------------- | ------- | --------- | -------- | ------- | ------------ | ------- | ------- | ---- |
| Level   | Method           | Acc     | P         | R        | F1      | P            | R       | F1      | n    |
| Level 1 | Without GBI      | 45.14%  | 27.21%    | 28.91%   | 25.71%  | 44.79%       | 45.14%  | 43.72%  | 4484 |
| Level 1 | With GBI         | 44.54%  | 23.90%    | 20.25%   | 20.07%  | 44.07%       | 44.54%  | 42.29%  | 4484 |
|         | w/ GBI - w/o GBI | \-0.60% | \-3.31%   | \-8.66%  | \-5.65% | \-0.72%      | \-0.60% | \-1.43% |      |
|         |                  |         |           |          |         |              |         |         |      |
| Level 2 | Without GBI      | 48.46%  | 33.01%    | 32.81%   | 30.77%  | 46.48%       | 48.46%  | 45.56%  | 4484 |
| Level 2 | With GBI         | 48.44%  | 32.37%    | 21.44%   | 23.90%  | 46.16%       | 48.44%  | 45.50%  | 4484 |
|         | w/ GBI - w/o GBI | \-0.02% | \-0.64%   | \-11.36% | \-6.87% | \-0.32%      | \-0.02% | \-0.06% |      |
|         |                  |         |           |          |         |              |         |         |      |
| Level 3 | Without GBI      | 41.73%  | 21.77%    | 52.49%   | 28.62%  | 75.16%       | 41.73%  | 43.56%  | 4484 |
| Level 3 | With GBI         | 70.54%  | 33.11%    | 25.73%   | 26.39%  | 69.26%       | 70.54%  | 69.42%  | 4484 |
|         | w/ GBI - w/o GBI | 28.81%  | 11.34%    | \-26.76% | \-2.23% | \-5.90%      | 28.81%  | 25.86%  |      |
|         |                  |         |           |          |         |              |         |         |      |
| Level 4 | Without GBI      | 84.57%  | 15.75%    | 63.72%   | 17.36%  | 99.11%       | 84.57%  | 91.06%  | 4484 |
| Level 5 | With GBI         | 97.06%  | 24.52%    | 32.13%   | 25.15%  | 98.96%       | 97.06%  | 97.96%  | 4484 |
|         | w/ GBI - w/o GBI | 12.49%  | 8.78%     | \-31.58% | 7.80%   | \-0.15%      | 12.49%  | 6.90%   |