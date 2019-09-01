<h1 align="center">Welcome to capture-rdt 👋</h1>
<p>
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://play.google.com/store/apps/details?id=edu.washington.cs.ubicomplab.rdt_reader&amp;hl=en_US">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" target="_blank" />
  </a>
</p>

> Python implementation of the capture RDT version of University of Washington UbiComp Lab

### 🏠 [Homepage](https://github.com/medic/rdt-capture/tree/master/app)

## Install

```sh
#Need to have Python installed on computer before, check Python documentation for detail
brew install opencv
pip install numpy matplotlib
pip install opencv-python opencv-contrib-python

or 

pip install --user --requirement requirements.txt
```

To run the scraping script to store images on S3 storage, you need to provided a `cough_photos_key.txt` under
keys folder

## Usage

```sh
#To run the core Image processing algorithm on the CLI
python main.py

#To run desktop GUI application for testing the output images of RDT
python gui.py

# To run scraping functions, provide a path to the barcodes txt file
python scrape.py --barcodes input/barcodes.txt 
# or use the default:  python scrape.py

# To run algorithm report anaysis, provide a path to the csv file
python report.py --f resources/query.csv

```

## Author

👤 **Hung Ngo**

* Github: [@hungngo97](https://github.com/hungngo97)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_