# import the necessary packages
import subprocess
from ImageProcessorScrapeReport import ImageProcessorScrapeReport
import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='resources/query.csv',
                        help='URLs image path')
    parser.add_argument('--min', type=int, default=120,
                        help='Min variable to start')
    parser.add_argument('--max', type=int, default=180,
                        help='Max variable to stop')
    parser.add_argument('--step', type=int, default=10,
                        help='Step value between min and max')
    parser.add_argument('--db', type=str,
                        help='number of URLs to debug')
    # This url passed in should be a list of url ( like a text file)
    args = parser.parse_args()
    imgProc = ImageProcessorScrapeReport()

    tpr = []
    fpr = []
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists('log/rocPeakConstant'):
        os.makedirs('log/rocPeakConstant')
        print("Directory ", 'log/rocPeakConstant',  " Created ")
    else:
        print("Directory ", 'log/rocPeakConstant',  " already exists")

    # for i in range(args.min, args.max + 1, args.step):
    #     variables = {}
    #     with open('variables/variables.json', 'r+') as json_file:
    #         variables = json.load(json_file)
    #     variables['PEAK_HEIGHT_THRESHOLD'] = i

    #     with open('variables/variables.json', 'w+') as json_file:
    #         json.dump(variables, json_file)

    #     pythonResultStats = imgProc.processFile(args.f, args.db)
    #     tpr.append(
    #         pythonResultStats['python_expert_response']['truePositiveRate'])
    #     fpr.append(
    #         pythonResultStats['python_expert_response']['falsePositiveRate'])

    #     with open('log/rocPeakConstant/constant' + str(i) + '.json', 'w+') as json_file:
    #         json.dump(pythonResultStats, json_file)

    # with open('log/rocPeakConstant/' + 'summary' + '.json', 'w+') as json_file:
    #     json.dump({"tpr": tpr, "fpr": fpr}, json_file)

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    ident = [0.0, 1.0]
    inversefpr = np.array(map(lambda x: 1 - x, fpr))
    plt.figure(1)
    plt.scatter(tpr, fpr, marker='o')
    plt.plot(ident, ident)
    plt.savefig('log/rocPeakConstant/roc1.png')

    plt.figure(2)
    plt.scatter(fpr, tpr, marker='o')
    plt.plot(ident, ident)
    plt.savefig('log/rocPeakConstant/roc2.png')

    plt.figure(3)
    plt.scatter(tpr, inversefpr, marker='o')
    plt.plot(ident, ident)
    plt.savefig('log/rocPeakConstant/roc3.png')

    plt.figure(4)
    plt.scatter(tpr, fpr, marker='o')
    plt.plot(ident, ident)
    plt.savefig('log/rocPeakConstant/roc4.png')


if __name__ == '__main__':
    main()
