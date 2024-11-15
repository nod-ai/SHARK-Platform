import re

# Sample log content (replace this with the actual log content)
log = """
2024-11-14 08:28:10 - INFO - 127.0.0.1:46952 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:42.981] [info] [metrics.py:52] Completed inference process (batch size 1) in 5377ms
[2024-11-14 08:28:42.987] [info] [metrics.py:52] Completed inference process (batch size 1) in 5380ms
[2024-11-14 08:28:42.988] [info] [metrics.py:44] THROUGHPUT: 0.18569706706675862 samples per second
2024-11-14 08:28:42 - INFO - 127.0.0.1:52798 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:42.994] [info] [metrics.py:44] THROUGHPUT: 0.18557775139781596 samples per second
2024-11-14 08:28:42 - INFO - 127.0.0.1:58828 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:42.998] [info] [metrics.py:52] Completed inference process (batch size 1) in 5391ms
[2024-11-14 08:28:43.007] [info] [metrics.py:44] THROUGHPUT: 0.18516820146323487 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:52924 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.016] [info] [metrics.py:52] Completed inference process (batch size 1) in 5412ms
[2024-11-14 08:28:43.021] [info] [metrics.py:52] Completed inference process (batch size 1) in 5415ms
[2024-11-14 08:28:43.021] [info] [metrics.py:52] Completed inference process (batch size 1) in 5415ms
[2024-11-14 08:28:43.022] [info] [metrics.py:52] Completed inference process (batch size 1) in 5416ms
[2024-11-14 08:28:43.024] [info] [metrics.py:44] THROUGHPUT: 0.1844849708208067 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:47118 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.028] [info] [metrics.py:44] THROUGHPUT: 0.18440449381491747 samples per second
[2024-11-14 08:28:43.029] [info] [metrics.py:44] THROUGHPUT: 0.18438347365068028 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:44058 - "POST /generate HTTP/1.1" 200
2024-11-14 08:28:43 - INFO - 127.0.0.1:47276 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.030] [info] [metrics.py:52] Completed inference process (batch size 1) in 5424ms
[2024-11-14 08:28:43.030] [info] [metrics.py:52] Completed inference process (batch size 1) in 5420ms
[2024-11-14 08:28:43.030] [info] [metrics.py:52] Completed inference process (batch size 1) in 5424ms
[2024-11-14 08:28:43.035] [info] [metrics.py:44] THROUGHPUT: 0.18418199480551325 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:34574 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.038] [info] [metrics.py:44] THROUGHPUT: 0.18409785465610476 samples per second
[2024-11-14 08:28:43.038] [info] [metrics.py:44] THROUGHPUT: 0.18407806766972434 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:34780 - "POST /generate HTTP/1.1" 200
2024-11-14 08:28:43 - INFO - 127.0.0.1:48534 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.041] [info] [metrics.py:52] Completed inference process (batch size 1) in 5435ms
[2024-11-14 08:28:43.043] [info] [metrics.py:52] Completed inference process (batch size 1) in 5438ms
2024-11-14 08:28:43 - INFO - 127.0.0.1:32876 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.043] [info] [metrics.py:44] THROUGHPUT: 0.18390164351606034 samples per second
[2024-11-14 08:28:43.044] [info] [metrics.py:52] Completed inference process (batch size 1) in 5438ms
[2024-11-14 08:28:43.049] [info] [metrics.py:44] THROUGHPUT: 0.1837019318007174 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:51430 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.050] [info] [metrics.py:44] THROUGHPUT: 0.18360602805729265 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:47512 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.052] [info] [metrics.py:44] THROUGHPUT: 0.18361447572191944 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:52550 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.055] [info] [metrics.py:52] Completed inference process (batch size 1) in 5450ms
[2024-11-14 08:28:43.056] [info] [metrics.py:52] Completed inference process (batch size 1) in 5450ms
[2024-11-14 08:28:43.061] [info] [metrics.py:52] Completed inference process (batch size 1) in 5455ms
[2024-11-14 08:28:43.062] [info] [metrics.py:52] Completed inference process (batch size 1) in 5457ms
[2024-11-14 08:28:43.063] [info] [metrics.py:44] THROUGHPUT: 0.18322249190563877 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:41644 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.063] [info] [metrics.py:44] THROUGHPUT: 0.1832230761868371 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:35368 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.064] [info] [metrics.py:52] Completed inference process (batch size 1) in 5458ms
[2024-11-14 08:28:43.066] [info] [metrics.py:52] Completed inference process (batch size 1) in 5460ms
[2024-11-14 08:28:43.067] [info] [metrics.py:52] Completed inference process (batch size 1) in 5461ms
[2024-11-14 08:28:43.068] [info] [metrics.py:52] Completed inference process (batch size 1) in 5462ms
[2024-11-14 08:28:43.069] [info] [metrics.py:44] THROUGHPUT: 0.18304873369483057 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:57890 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.069] [info] [metrics.py:52] Completed inference process (batch size 1) in 5464ms
[2024-11-14 08:28:43.070] [info] [metrics.py:44] THROUGHPUT: 0.18299370842201312 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:50860 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.073] [info] [metrics.py:44] THROUGHPUT: 0.18288441095343283 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:35844 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.074] [info] [metrics.py:44] THROUGHPUT: 0.1828765087471988 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:48410 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.075] [info] [metrics.py:44] THROUGHPUT: 0.18282233602724648 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:51704 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.075] [info] [metrics.py:52] Completed inference process (batch size 1) in 5469ms
[2024-11-14 08:28:43.076] [info] [metrics.py:44] THROUGHPUT: 0.18280612074563105 samples per second
[2024-11-14 08:28:43.076] [info] [metrics.py:52] Completed inference process (batch size 1) in 5470ms
2024-11-14 08:28:43 - INFO - 127.0.0.1:46376 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.077] [info] [metrics.py:44] THROUGHPUT: 0.18276756622403867 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:45808 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.083] [info] [metrics.py:52] Completed inference process (batch size 1) in 5477ms
[2024-11-14 08:28:43.083] [info] [metrics.py:44] THROUGHPUT: 0.1825813477200838 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:35324 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.084] [info] [metrics.py:44] THROUGHPUT: 0.18255683171164483 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:43534 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.085] [info] [metrics.py:52] Completed inference process (batch size 1) in 5479ms
[2024-11-14 08:28:43.088] [info] [metrics.py:52] Completed inference process (batch size 1) in 5482ms
[2024-11-14 08:28:43.091] [info] [metrics.py:44] THROUGHPUT: 0.18229083772866955 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:58602 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.093] [info] [metrics.py:44] THROUGHPUT: 0.1822533000816865 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:43160 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.096] [info] [metrics.py:44] THROUGHPUT: 0.18215596753811167 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:36718 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.100] [info] [metrics.py:52] Completed inference process (batch size 1) in 5493ms
[2024-11-14 08:28:43.103] [info] [metrics.py:52] Completed inference process (batch size 1) in 5497ms
[2024-11-14 08:28:43.107] [info] [metrics.py:44] THROUGHPUT: 0.18176592603070366 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:42562 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.108] [info] [metrics.py:52] Completed inference process (batch size 1) in 5502ms
[2024-11-14 08:28:43.110] [info] [metrics.py:44] THROUGHPUT: 0.18164378744453424 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:53384 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.114] [info] [metrics.py:52] Completed inference process (batch size 1) in 5508ms
[2024-11-14 08:28:43.116] [info] [metrics.py:44] THROUGHPUT: 0.18147876473753666 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:43498 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.117] [info] [metrics.py:52] Completed inference process (batch size 1) in 5511ms
[2024-11-14 08:28:43.118] [info] [metrics.py:52] Completed inference process (batch size 1) in 5512ms
[2024-11-14 08:28:43.118] [info] [metrics.py:52] Completed inference process (batch size 1) in 5512ms
[2024-11-14 08:28:43.119] [info] [metrics.py:52] Completed inference process (batch size 1) in 5513ms
[2024-11-14 08:28:43.121] [info] [metrics.py:52] Completed inference process (batch size 1) in 5515ms
[2024-11-14 08:28:43.121] [info] [metrics.py:44] THROUGHPUT: 0.1812948719832844 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:39060 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.122] [info] [metrics.py:52] Completed inference process (batch size 1) in 5516ms
[2024-11-14 08:28:43.122] [info] [metrics.py:52] Completed inference process (batch size 1) in 5516ms
[2024-11-14 08:28:43.125] [info] [metrics.py:44] THROUGHPUT: 0.1811743984314216 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:43678 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.126] [info] [metrics.py:44] THROUGHPUT: 0.1811377963130895 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:40786 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.126] [info] [metrics.py:44] THROUGHPUT: 0.18114753613370121 samples per second
[2024-11-14 08:28:43.126] [info] [metrics.py:44] THROUGHPUT: 0.18110297622075242 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:42492 - "POST /generate HTTP/1.1" 200
2024-11-14 08:28:43 - INFO - 127.0.0.1:39684 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.128] [info] [metrics.py:52] Completed inference process (batch size 1) in 5522ms
[2024-11-14 08:28:43.129] [info] [metrics.py:44] THROUGHPUT: 0.18105860235825336 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:56536 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.130] [info] [metrics.py:44] THROUGHPUT: 0.18103122743612973 samples per second
[2024-11-14 08:28:43.130] [info] [metrics.py:52] Completed inference process (batch size 1) in 5523ms
[2024-11-14 08:28:43.130] [info] [metrics.py:44] THROUGHPUT: 0.18101518768757818 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:52828 - "POST /generate HTTP/1.1" 200
2024-11-14 08:28:43 - INFO - 127.0.0.1:47484 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.131] [info] [metrics.py:52] Completed inference process (batch size 1) in 5524ms
[2024-11-14 08:28:43.132] [info] [metrics.py:52] Completed inference process (batch size 1) in 5526ms
[2024-11-14 08:28:43.135] [info] [metrics.py:52] Completed inference process (batch size 1) in 5529ms
[2024-11-14 08:28:43.135] [info] [metrics.py:44] THROUGHPUT: 0.18084154207764283 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:33822 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.137] [info] [metrics.py:44] THROUGHPUT: 0.18078267708532725 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:39654 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.138] [info] [metrics.py:52] Completed inference process (batch size 1) in 5532ms
[2024-11-14 08:28:43.140] [info] [metrics.py:44] THROUGHPUT: 0.18068830188354118 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:50096 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.141] [info] [metrics.py:52] Completed inference process (batch size 1) in 5535ms
[2024-11-14 08:28:43.141] [info] [metrics.py:44] THROUGHPUT: 0.18066654835602772 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:48434 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.142] [info] [metrics.py:44] THROUGHPUT: 0.18061058142832548 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:53676 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.146] [info] [metrics.py:44] THROUGHPUT: 0.18049809905468714 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:54946 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.147] [info] [metrics.py:52] Completed inference process (batch size 1) in 5543ms
[2024-11-14 08:28:43.148] [info] [metrics.py:52] Completed inference process (batch size 1) in 5542ms
[2024-11-14 08:28:43.149] [info] [metrics.py:44] THROUGHPUT: 0.18039276236180407 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:47472 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.150] [info] [metrics.py:52] Completed inference process (batch size 1) in 5545ms
[2024-11-14 08:28:43.154] [info] [metrics.py:52] Completed inference process (batch size 1) in 5548ms
[2024-11-14 08:28:43.155] [info] [metrics.py:44] THROUGHPUT: 0.18014040570399836 samples per second
[2024-11-14 08:28:43.155] [info] [metrics.py:52] Completed inference process (batch size 1) in 5549ms
2024-11-14 08:28:43 - INFO - 127.0.0.1:36260 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.156] [info] [metrics.py:52] Completed inference process (batch size 1) in 5549ms
[2024-11-14 08:28:43.156] [info] [metrics.py:44] THROUGHPUT: 0.1801768380088354 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:59420 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.158] [info] [metrics.py:44] THROUGHPUT: 0.18008399876002648 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:34862 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.161] [info] [metrics.py:44] THROUGHPUT: 0.1799940280988335 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:38936 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.163] [info] [metrics.py:44] THROUGHPUT: 0.17994180415693786 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:55318 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.163] [info] [metrics.py:44] THROUGHPUT: 0.17994984851457532 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:35910 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.164] [info] [metrics.py:52] Completed inference process (batch size 1) in 5559ms
[2024-11-14 08:28:43.164] [info] [metrics.py:52] Completed inference process (batch size 1) in 5558ms
[2024-11-14 08:28:43.166] [info] [metrics.py:52] Completed inference process (batch size 1) in 5559ms
[2024-11-14 08:28:43.169] [info] [metrics.py:52] Completed inference process (batch size 1) in 5565ms
[2024-11-14 08:28:43.171] [info] [metrics.py:44] THROUGHPUT: 0.17963241922020085 samples per second
[2024-11-14 08:28:43.171] [info] [metrics.py:44] THROUGHPUT: 0.17966916197650198 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:57516 - "POST /generate HTTP/1.1" 200
2024-11-14 08:28:43 - INFO - 127.0.0.1:35598 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.173] [info] [metrics.py:44] THROUGHPUT: 0.17961581873289753 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:57634 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.177] [info] [metrics.py:44] THROUGHPUT: 0.17943001517387067 samples per second
[2024-11-14 08:28:43.177] [info] [metrics.py:52] Completed inference process (batch size 1) in 5571ms
2024-11-14 08:28:43 - INFO - 127.0.0.1:58806 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.178] [info] [metrics.py:52] Completed inference process (batch size 1) in 5572ms
[2024-11-14 08:28:43.184] [info] [metrics.py:52] Completed inference process (batch size 1) in 5578ms
[2024-11-14 08:28:43.184] [info] [metrics.py:44] THROUGHPUT: 0.17923159744543832 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:37182 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.185] [info] [metrics.py:44] THROUGHPUT: 0.1792203088638572 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:37044 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.189] [info] [metrics.py:52] Completed inference process (batch size 1) in 5582ms
[2024-11-14 08:28:43.191] [info] [metrics.py:44] THROUGHPUT: 0.17903126434461933 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:41130 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.196] [info] [metrics.py:44] THROUGHPUT: 0.1788782451584508 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:51710 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.208] [info] [metrics.py:52] Completed inference process (batch size 1) in 5602ms
[2024-11-14 08:28:43.215] [info] [metrics.py:44] THROUGHPUT: 0.17826016830637276 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:46532 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.216] [info] [metrics.py:52] Completed inference process (batch size 1) in 5610ms
[2024-11-14 08:28:43.224] [info] [metrics.py:44] THROUGHPUT: 0.17800234645598784 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:54432 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.227] [info] [metrics.py:52] Completed inference process (batch size 1) in 5622ms
2024-11-14 08:28:43 - INFO - 127.0.0.1:53380 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.235] [info] [metrics.py:44] THROUGHPUT: 0.1776107813533037 samples per second
[2024-11-14 08:28:43.241] [info] [metrics.py:52] Completed inference process (batch size 1) in 5637ms
[2024-11-14 08:28:43.242] [info] [metrics.py:52] Completed inference process (batch size 1) in 5635ms
[2024-11-14 08:28:43.249] [info] [metrics.py:44] THROUGHPUT: 0.17716528631514256 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:51812 - "POST /generate HTTP/1.1" 200
[2024-11-14 08:28:43.249] [info] [metrics.py:44] THROUGHPUT: 0.17719879571008898 samples per second
2024-11-14 08:28:43 - INFO - 127.0.0.1:59630 - "POST /generate HTTP/1.1" 200
"""

# Regex patterns for throughput and inference time
throughput_pattern = r"THROUGHPUT: ([\d.]+) samples per second"
inference_pattern = r"inference process.*in (\d+)ms"

throughput_values = [float(x) for x in re.findall(throughput_pattern, log)]
inference_times = [int(x) for x in re.findall(inference_pattern, log)]

# Calculate total throughput
if throughput_values:
    total_time = max(inference_times)
    throughput = len(inference_times)/(total_time/1000)
    print(f"Total Throughput: {throughput:.4f} samples per second")
    print(f"Num of Samples: {len(inference_times)}")
else:
    print("No throughput data found.")

# Calculate average inference time
if inference_times:
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
else:
    print("No inference time data found.")
