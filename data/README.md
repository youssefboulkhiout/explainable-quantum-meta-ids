# Datasets

The CSV files are not distributed with this repository. Download them from
their original sources and save them here with these exact names:

| File | Dataset | Label column | Source |
|---|---|---|---|
| `ASEADOS_SDN_IoT.csv` | ASEADOS-SDN-IoT | `label` / `class` / `attack_type` / `category` (auto-detected) | Yasarathna & Le-Khac, *Internet of Things* (2026), doi:10.1016/j.iot.2026.101891 |
| `RT-IoT2022.csv` | RT-IoT2022 | `Attack_type` | UCI Machine Learning Repository; Sharmila & Nagapadma, *Cybersecurity* 6:41 (2023), doi:10.1186/s42400-023-00178-5 |
| `UNSW-NB15.csv` | UNSW-NB15 | `attack_cat` | UNSW Canberra Cyber; Moustafa & Slay, MilCIS 2015, doi:10.1109/MilCIS.2015.7348942 |

Paths and label candidates are set in `xqmids/config.py` (`DATASETS`). The
label column is detected automatically (case-insensitive), identifier
columns are dropped, and rows with NaN/inf values and duplicates are removed
by the pipeline itself, so the raw CSV files can be used as downloaded.
