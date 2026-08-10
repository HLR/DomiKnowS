This is the Multi-Axis Temporal RElations for Start-points (i.e., `MATRES`) dataset, collected via the annotation scheme proposed in 

```
@inproceedings{NingWuRo18,
    author = {Qiang Ning and Hao Wu and Dan Roth},
    title = {A Multi-Axis Annotation Scheme for Event Temporal Relations},
    booktitle = {ACL},
    month = {7},
    year = {2018},
    url = "http://cogcomp.org/papers/NingWuRo18.pdf",
}
```

The dataset is formatted as follows.
```
docid, verb1, verb2, eiid1, eiid2, relation
```
The `eiid`s are the same with those in TempEval3. Although in the paper above we had only annotated TimeBank-Dense (36 documents), we later extended to the entire TempEval3 dataset. According to TempEval3, the dataset is split into three parts: TimeBank, AQUAINT, and Platinum, each of which has a separate file in this repo.

Note `eiid` is not `eid`. These two id numbers are sometimes inconsistent in TempEval3, so please make sure you're using `eiid` instead of `eid`.

A few remarks:
- MATRES is only for verb events, so those nominal events in the original TempEval3 won't appear in MATRES.
- In this repo, we only include relations among those events that are on the main-axis. Please refer to the paper for the definition of main-axis.
- For other axes and relations, please refer to this [readme file](https://github.com/qiangning/MATRES/tree/master/rawdata).

### Update Jan 2022
Some users have reported to us that the TempEval3 dataset is no longer available online. I do have a copy of TempEval3 from my own project in the past: https://github.com/qiangning/StructTempRel-EMNLP17/tree/master/data/TempEval3
