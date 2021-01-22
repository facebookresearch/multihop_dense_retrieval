## Evaluating retrieval:
```
Evaluating 7405 samples...
	Avg PR: 0.8428089128966915
	Avg P-EM: 0.6592842673869007
	Avg 1-Recall: 0.7906819716407832
	Path Recall: 0.6592842673869007
comparison Questions num: 1487
	Avg PR: 0.9932750504371217
	Avg P-EM: 0.9482178883658372
	Avg 1-Recall: 0.9643577673167452
	Path Recall: 0.9482178883658372
bridge Questions num: 5918
	Avg PR: 0.805001689760054
	Avg P-EM: 0.5866846907739101
	Avg 1-Recall: 0.7470429199053734
	Path Recall: 0.5866846907739101
```

## Evaluating QA
```
01/21/2021 17:01:49 - INFO - __main__ - evaluated 7405 questions...
01/21/2021 17:01:49 - INFO - __main__ - chain ranking em: 0.8113436866981769
01/21/2021 17:01:50 - INFO - __main__ - .......Using combination factor 0.8......
01/21/2021 17:01:50 - INFO - __main__ - answer em: 0.6233625928426739, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - answer f1: 0.7504594111976622, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - sp em: 0.5654287643484133, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - sp f1: 0.7942837708469039, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - joint em: 0.42052667116812964, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - joint f1: 0.6631669237532106, count: 7405
01/21/2021 17:01:50 - INFO - __main__ - Best joint F1 from combination 0.7504594111976622
01/21/2021 17:01:51 - INFO - __main__ - test performance {'em': 0.6233625928426739, 'f1': 0.7504594111976622, 'joint_em': 0.42052667116812964, 'joint_f1': 0.6631669237532106, 'sp_em': 0.5654287643484133, 'sp_f1': 0.7942837708469039}
```
