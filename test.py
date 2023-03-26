

import csv


with open('log/log.csv', 'a') as csvfile:
    wt = csv.DictWriter(csvfile, fieldnames=['acc', 'macc', 'miou', 'average'])

    wt.writeheader()
    wt.writerow({'acc': 1, 'macc': 2, 'miou': 4, 'average': (1 + 3) / 2})