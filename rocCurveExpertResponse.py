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

"""
def process_images(payload):
    # display the process ID for debugging and initialize the hashes
    # dictionary
    print("[INFO] starting process {}".format(payload["id"]))
    hashes = {}

    variables = {}
    variables['PEAK_HEIGHT_THRESHOLD'] = payload['threshold']

    with open(payload['outputPath'] + 'variables/variables.json', 'w+') as json_file:
        json.dump(variables, json_file)
    reload(ImageProcessorScrapeReport)
    imgProc = ImageProcessorScrapeReport.ImageProcessorScrapeReport(payload['outputPath'])
    pythonResultStats = imgProc.processFile(payload["file"], payload["db"])

    with open(payload['outputPath'] + 'log/rocPeakConstant/result.json', 'w+') as json_file:
        json.dump(pythonResultStats, json_file)


"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str,
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
    currentProcID = 0

    processing with OpenCV and PythonPython
	# initialize the list of payloads
	payloads = []

    # threshold List
    thresholds = list(args.min + args.step * i for i in procIDs) #TODO: double check this

	# loop over the set chunked image paths
	for i in procIDs:
		# construct the path to the output intermediary file for the
		# current process
		outputPath = 'CPU' + str(i) + '/'

		# construct a dictionary of data for the payload, then add it
		# to the payloads list
		data = {
			"id": i,
			"file": args.f,
            "db": args.db,
			"output_path": outputPath,
            "threshold": thresholds[i]
		}
		payloads.append(data)

    # construct and launch the processing pool
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)

    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")

    # Reduce Collecting Phase

    """

    tpr = []
    fpr = []
    thresholds = None
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists('log/rocPeakConstant'):
        os.makedirs('log/rocPeakConstant')
        print("Directory ", 'log/rocPeakConstant',  " Created ")
    else:
        print("Directory ", 'log/rocPeakConstant',  " already exists")

    if (not args.f):
        thresholds = []
        for file in os.listdir("log/rocPeakConstant/"):
            if file.endswith(".json") and file.startswith("constant"):
                print('found file ', file)
                result = {}
                with open("log/rocPeakConstant/" + file, 'r') as json_file:
                    result = json.load(json_file)
                if result['python_expert_response']['truePositiveRate'] == 'N/A' or \
                    result['python_expert_response']['falsePositiveRate'] == 'N/A':
                    continue
                tpr.append(result['python_expert_response']['truePositiveRate'])
                fpr.append(result['python_expert_response']['falsePositiveRate'])
                threshold = int(file.split('.')[0].split('constant')[1])
                print('with threshold ', threshold)
                thresholds.append(threshold)
    else:    
        # Load from previous checkpoint if have it
        start = args.min
        for i in range(args.min, args.max + 1, args.step):
            thresholdFilePath = 'log/rocPeakConstant/' + 'constant' + str(i) + '.json'
            if os.path.exists(thresholdFilePath):
                with open(thresholdFilePath, 'r') as json_file:
                    result = json.load(json_file)
                    print('FOUND result constant ' + str(i), result)
                    tpr.append(result['python_expert_response']['truePositiveRate'])
                    fpr.append(result['python_expert_response']['falsePositiveRate'])
                start = i + args.step
            else:
                break # start from where it does not have cached result

        for i in range(start, args.max + 1, args.step):
            variables = {}
            with open('variables/variables.json', 'r+') as json_file:
                variables = json.load(json_file)
            variables['PEAK_HEIGHT_THRESHOLD'] = i

            with open('variables/variables.json', 'w+') as json_file:
                json.dump(variables, json_file)
            imgProc = ImageProcessorScrapeReport.ImageProcessorScrapeReport('')
            pythonResultStats = imgProc.processFile(args.f, args.db)
            tpr.append(
                pythonResultStats['python_expert_response']['truePositiveRate'])
            fpr.append(
                pythonResultStats['python_expert_response']['falsePositiveRate'])

            with open('log/rocPeakConstant/constant' + str(i) + '.json', 'w+') as json_file:
                json.dump(pythonResultStats, json_file)

        with open('log/rocPeakConstant/' + 'summary' + '.json', 'w+') as json_file:
            json.dump({"tpr": tpr, "fpr": fpr}, json_file)

    # tpr = np.array([0.5,0.7,0.8])
    # fpr = np.array([0.4,0.6,0.9])
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

    fig = plt.figure(5)
    ax1 = fig.add_subplot(111)
    thresholds = np.arange(args.min, args.max + 1, args.step) if thresholds is None else np.array(thresholds)

    print("TPR", tpr)
    print("FPR", fpr)
    print("THRESHOLDS", thresholds)
    ax1.scatter(thresholds, tpr, s=10, c='b', marker="s", label='TPR')
    ax1.scatter(thresholds, fpr, s=10, c='r', marker="o", label='FPR')
    plt.legend(loc='best');
    plt.xlabel('Threshold')
    plt.savefig('log/rocPeakConstant/thresholdvsTPRvsFPR.png')
    plt.show()



if __name__ == '__main__':
    main()
