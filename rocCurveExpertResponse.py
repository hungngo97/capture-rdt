# import the necessary packages
import subprocess
import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import ImageProcessorScrapeReport


def process_images(payload):
    # display the process ID for debugging and initialize the hashes
    # dictionary
    print("[INFO] starting process {}".format(payload["id"]))
    hashes = {}

    # loop over the image paths
    for imagePath in payload["input_paths"]:
        # load the input image, compute the hash, and conver it
        image = cv2.imread(imagePath)
        h = dhash(image)
        h = convert_hash(h)

        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(imagePath)
        hashes[h] = l

    # serialize the hashes dictionary to disk using the supplied
    # output path
    print("[INFO] process {} serializing hashes".format(payload["id"]))
    f = open(payload["output_path"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()


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

    """
    # determine the number of concurrent processes to launch when
    # distributing the load across the system, then create the list
    # of process IDs
    procs = args["procs"] if args["procs"] > 0 else cpu_count()
    procIDs = list(range(0, procs))

    """

    tpr = []
    fpr = []
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists('log/rocPeakConstant'):
        os.makedirs('log/rocPeakConstant')
        print("Directory ", 'log/rocPeakConstant',  " Created ")
    else:
        print("Directory ", 'log/rocPeakConstant',  " already exists")

    for i in range(args.min, args.max + 1, args.step):
        variables = {}
        with open('variables/variables.json', 'r+') as json_file:
            variables = json.load(json_file)
        variables['PEAK_HEIGHT_THRESHOLD'] = i

        with open('variables/variables.json', 'w+') as json_file:
            json.dump(variables, json_file)
        reload(ImageProcessorScrapeReport)
        imgProc = ImageProcessorScrapeReport.ImageProcessorScrapeReport()
        pythonResultStats = imgProc.processFile(args.f, args.db)
        tpr.append(
            pythonResultStats['python_expert_response']['truePositiveRate'])
        fpr.append(
            pythonResultStats['python_expert_response']['falsePositiveRate'])

        with open('log/rocPeakConstant/constant' + str(i) + '.json', 'w+') as json_file:
            json.dump(pythonResultStats, json_file)

    with open('log/rocPeakConstant/' + 'summary' + '.json', 'w+') as json_file:
        json.dump({"tpr": tpr, "fpr": fpr}, json_file)

    threshold = np.array(range(args.min, args.max + 1, args.step))
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    ident = [0.0, 1.0]
    inversefpr = np.array(map(lambda x: 1 - x, fpr))

    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 13,
    }

    plt.figure(1)
    plt.scatter(tpr, fpr, marker='o')
    plt.plot(ident, ident, '--', linewidth=2)
    plt.title('TPR vs FPR' , fontdict=font)
    plt.xlabel('TPR ', fontdict=font)
    plt.ylabel('FPR', fontdict=font)
    plt.savefig('log/rocPeakConstant/TPRvsFPR.png')

    plt.figure(2)
    plt.scatter(fpr, tpr, marker='o')
    plt.plot(ident, ident, '--', linewidth=2)
    plt.title('FPR vs TPR' , fontdict=font)
    plt.xlabel('FPR ', fontdict=font)
    plt.ylabel('TPR', fontdict=font)
    plt.savefig('log/rocPeakConstant/FPRvsTPR.png')

    plt.figure(3)
    plt.scatter(tpr, inversefpr, marker='o')
    plt.plot(ident, ident, '--', linewidth=2)
    plt.title('TPR vs 1 - FPR' , fontdict=font)
    plt.xlabel('TPR ', fontdict=font)
    plt.ylabel('1 - FPR', fontdict=font)
    plt.savefig('log/rocPeakConstant/TPRvsInverseFPR.png')

    plt.figure(4)
    plt.scatter(inversefpr, tpr, marker='o')
    plt.plot(ident, ident,'--', linewidth=2)
    plt.title('1 - FPR vs TPR' , fontdict=font)
    plt.xlabel('1 - FPR ', fontdict=font)
    plt.ylabel('TPR', fontdict=font)
    plt.savefig('log/rocPeakConstant/inverseFPRvsTPR.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(threshold, tpr, s=10, c='b', marker="s", label='TPR')
    ax1.scatter(threshold, fpr, s=10, c='r', marker="o", label='FPR')
    plt.legend(loc='lower right');
    plt.show()
    plt.savefig('log/rocPeakConstant/thresholdvsTPRvsFPR.png')



if __name__ == '__main__':
    main()
