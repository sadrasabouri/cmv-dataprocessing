# Data with Delta Information
Run code modules in order of the prefix number. The number shows the level of 'i'-th handed data the process is being done on.

## Commands to run
```
python 1_make_dataset.py changemyview_submissions_df-head10.jsonl changemyview_comments_df-head100.jsonl ../deltas.csv cmv_delta.jsonl
```

## Stats:

### Frequency of Deltas on Comments
Out of ~93.6k unique comments over 47k posts with at least one delta we saw a power-law on the distribution of deltas on comments:
```
$ sed -n "s/.*('\([^']*\)',[[:space:]]*'[^']*').*/'\1'/p" deltas_dict.json | sort | uniq | wc -l
   47272
$ wc -l deltas_dict.json 
   93583 deltas_dict.json
$ sed -n "s/.*'count':[[:space:]]*\([0-9][0-9]*\).*/\1/p" deltas_dict.json | sort | uniq -c | sort -r
92944 1
 532 2
  57 3
  14 4
  12 5
   9 6
   5 15
   2 18
   2 12
   1 9
   1 8
   1 7
   1 22
   1 21
```
