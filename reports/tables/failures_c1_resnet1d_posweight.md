# Failure analysis: c1_resnet1d_posweight (seed0)

## Errors per class (frozen val thresholds)

| class   |   positives |   false_negatives |   false_positives |   threshold |   FN rate |
|:--------|------------:|------------------:|------------------:|------------:|----------:|
| NORM    |         963 |               107 |               171 |       0.593 |     0.111 |
| MI      |         550 |                93 |               182 |       0.469 |     0.169 |
| STTC    |         521 |                89 |               181 |       0.577 |     0.171 |
| CD      |         496 |               135 |                84 |       0.764 |     0.272 |
| HYP     |         262 |               114 |                89 |       0.808 |     0.435 |

## Sliced performance

| slice                 | group   |    n |   macro AUROC |   exact-match error |
|:----------------------|:--------|-----:|--------------:|--------------------:|
| metadata noise flag   | clean   | 1645 |        0.9342 |               0.388 |
| metadata noise flag   | flagged |  513 |        0.9298 |               0.402 |
| age band              | 40-60   |  654 |        0.9266 |               0.321 |
| age band              | 60-75   |  708 |        0.9275 |               0.434 |
| age band              | 75+     |  486 |        0.9107 |               0.535 |
| age band              | <40     |  310 |        0.9343 |               0.216 |
| sex (0=male,1=female) | 0       | 1110 |        0.9407 |               0.378 |
| sex (0=male,1=female) | 1       | 1048 |        0.925  |               0.405 |
| # superclass labels   | 1       | 1650 |        0.9291 |               0.315 |
| # superclass labels   | 2       |  398 |        0.8904 |               0.658 |
| # superclass labels   | 3       |  110 |        0.903  |               0.564 |

## Most confident misses

**NORM missed**

|   ecg_id | true labels   |   p(NORM) | noisy   |
|---------:|:--------------|----------:|:--------|
|    10887 | NORM          |     0.03  | False   |
|     1186 | NORM          |     0.051 | False   |
|    12711 | NORM          |     0.057 | False   |
|     1220 | CD|NORM       |     0.078 | False   |
|    12033 | NORM          |     0.084 | True    |
|      245 | NORM          |     0.091 | True    |
|     3486 | CD|HYP|NORM   |     0.093 | False   |
|    18416 | NORM          |     0.135 | False   |

**NORM false alarm**

|   ecg_id | true labels   |   p(NORM) | noisy   |
|---------:|:--------------|----------:|:--------|
|     7895 | MI            |     0.991 | False   |
|     9299 | HYP           |     0.99  | False   |
|      378 | CD            |     0.988 | True    |
|    20369 | HYP           |     0.986 | False   |
|     5337 | CD            |     0.985 | False   |
|     1745 | HYP           |     0.982 | False   |
|     1447 | HYP|MI        |     0.98  | True    |
|    16712 | MI            |     0.979 | False   |

**MI missed**

|   ecg_id | true labels   |   p(MI) | noisy   |
|---------:|:--------------|--------:|:--------|
|     7895 | MI            |   0.011 | False   |
|     1447 | HYP|MI        |   0.02  | True    |
|    15024 | MI|STTC       |   0.041 | True    |
|     1753 | MI            |   0.041 | False   |
|    19935 | CD|MI|STTC    |   0.058 | False   |
|    16712 | MI            |   0.062 | False   |
|     5777 | MI            |   0.067 | False   |
|    17868 | HYP|MI        |   0.067 | False   |

**MI false alarm**

|   ecg_id | true labels   |   p(MI) | noisy   |
|---------:|:--------------|--------:|:--------|
|     2417 | HYP           |   0.996 | False   |
|    15858 | STTC          |   0.985 | False   |
|    15526 | CD            |   0.983 | False   |
|    10887 | NORM          |   0.977 | False   |
|     1736 | STTC          |   0.956 | False   |
|     6776 | NORM          |   0.955 | False   |
|      172 | CD            |   0.955 | False   |
|    16700 | CD            |   0.948 | False   |

**STTC missed**

|   ecg_id | true labels   |   p(STTC) | noisy   |
|---------:|:--------------|----------:|:--------|
|    11132 | HYP|STTC      |     0.013 | False   |
|    18019 | MI|STTC       |     0.035 | False   |
|    17404 | MI|STTC       |     0.038 | False   |
|    15259 | MI|STTC       |     0.039 | False   |
|     1268 | STTC          |     0.051 | True    |
|      861 | STTC          |     0.059 | False   |
|     7919 | CD|HYP|STTC   |     0.06  | False   |
|     4618 | CD|MI|STTC    |     0.061 | False   |

**STTC false alarm**

|   ecg_id | true labels   |   p(STTC) | noisy   |
|---------:|:--------------|----------:|:--------|
|     9458 | CD            |     0.991 | False   |
|    16182 | HYP           |     0.989 | False   |
|    16782 | MI            |     0.984 | True    |
|    15141 | MI            |     0.983 | True    |
|     9144 | CD|MI         |     0.982 | False   |
|     8178 | CD|MI         |     0.979 | False   |
|     5911 | HYP           |     0.978 | False   |
|    12711 | NORM          |     0.977 | False   |

**CD missed**

|   ecg_id | true labels   |   p(CD) | noisy   |
|---------:|:--------------|--------:|:--------|
|    19461 | CD|NORM       |   0.007 | False   |
|     2909 | CD|NORM       |   0.011 | False   |
|    17888 | CD|NORM       |   0.012 | False   |
|      378 | CD            |   0.016 | True    |
|     5337 | CD            |   0.019 | False   |
|    21011 | CD            |   0.023 | True    |
|    13314 | CD|NORM       |   0.031 | False   |
|    18974 | CD            |   0.038 | False   |

**CD false alarm**

|   ecg_id | true labels   |   p(CD) | noisy   |
|---------:|:--------------|--------:|:--------|
|     7876 | HYP|MI|STTC   |   0.998 | False   |
|    13577 | MI|STTC       |   0.992 | False   |
|    15647 | MI            |   0.979 | False   |
|     8727 | NORM          |   0.974 | False   |
|    12033 | NORM          |   0.972 | True    |
|    12122 | HYP           |   0.971 | False   |
|     4614 | HYP|MI|STTC   |   0.967 | False   |
|    14314 | MI            |   0.964 | False   |

**HYP missed**

|   ecg_id | true labels   |   p(HYP) | noisy   |
|---------:|:--------------|---------:|:--------|
|     4167 | CD|HYP|NORM   |    0.004 | False   |
|    11872 | HYP|MI        |    0.005 | False   |
|     5289 | HYP|STTC      |    0.042 | False   |
|    20369 | HYP           |    0.042 | False   |
|     1447 | HYP|MI        |    0.043 | True    |
|     8501 | HYP|STTC      |    0.046 | False   |
|     1745 | HYP           |    0.052 | False   |
|     9299 | HYP           |    0.071 | False   |

**HYP false alarm**

|   ecg_id | true labels   |   p(HYP) | noisy   |
|---------:|:--------------|---------:|:--------|
|    20224 | CD|MI         |    0.992 | False   |
|    12178 | CD            |    0.989 | False   |
|    13007 | STTC          |    0.985 | False   |
|    16920 | CD            |    0.985 | True    |
|     6592 | CD|STTC       |    0.984 | False   |
|    14937 | CD|MI         |    0.983 | False   |
|    20120 | CD            |    0.981 | False   |
|    11896 | MI            |    0.98  | True    |
