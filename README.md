[![Construction image Docker](https://github.com/yann-Choho/Colorizing-puzzle-app-improved/actions/workflows/prod.yaml/badge.svg?branch=pierreBranch)](https://github.com/yann-Choho/Colorizing-puzzle-app-improved/actions/workflows/prod.yaml)
# ENSAE Paris | Institut Polytechnique de Paris

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/900px-LOGO-ENSAE.png" width="300">

## Course : Mise en production des projets de data-science
### Lino GALIANA and Romain AVOUAC

### Academic year: 2023-2024
February 2024 - May 2024


## Topic : Machine Learning-Driven Image Colorization, a Puzzle Quest to Get Your Image Colorful 🎨

### Realised by :

* Yann Eric CHOHO
* Thomas PIQUÉ
* Pierre REGNARD

### Based on a former project for the course "Infrastructures et systèmes logiciels" with Antoine CHANCEL, realised by :

* Rayan TALATE
* Yseult MASSON
* Yann Eric CHOHO
* Thomas PIQUÉ
* Pierre REGNARD


## Introduction

This application allows you to solve a puzzle of a black-and-white image. Once the puzzle is finished, the image is colorized by a Deep Learning model and appears on your screen!


## How to run the app
You don't need to run the underlying model to play with the puzzle, the app is available with this link https://puzzle.kub.sspcloud.fr/.

If you want to run it manually on your computer, then follow the step bellow :

- Option 1 : Clone this repository, then install the requirements library
```python
pip install -r requirements.txt
```
and launch app.py

- option 2 :  follow the step describe in option 1, but instead of directly launching app.py, Run with Docker and have fun !
  
First, build the docker image by running the following command in a terminal :

```bash
docker build . -t puzzle_image
```

Then, run the container:

```bash
docker run -p 8000:8000 puzzle_image
```

The app runs on `http://0.0.0.0:8000/`. If this url does not work, try `http://127.0.0.1:8000/` or `http://localhost:8000/`. If the colored image doesn't load after completing the puzzle, try using another browser.

- Option 3 : Instead of cloning the repo, you can upload the image directly from docker hub  and launch your container based on this image.
```bash
docker pull yannchoho/puzzle-colorization
```
## Description of the puzzle app

On the menu page, you can choose between 2 types of puzzle : sliding puzzle and free movement puzzle.

Free movement puzzle :
- the grid is empty
- the available pieces are below
- to place a piece, click first on the piece below and then on a cell
- click on a non-empty cell to remove a piece
- shuffle button will shuffle the available pieces
- when all the "picture" pieces are in the right place, all cells display their colored correspondant piece

Sliding puzzle : 
- one cell is "empty" and adjacent cells can swap position with that empty cell when clicked
- when all the "picture" pieces are in the right place, all cells display their colored correspondant piece
- a shuffle button on the right side of the puzzle ( ! some initial combinations are unsolvable ! )

Extra Features
- Add your own gray-and-white image to be colorised by our *Deep Learning Model*
- Choose your favorite image to play in a gallery of 20 images made available for you


## Scripts behind the app : organization of the repository

To develop the app, we followed several steps.
* **Scrap images** : to obtain enough images to train the model, we scrapped images on Google Image. The scripts for this part are available in the folder `scraping_images`. `scraping.py` does the scraping, and `color_to_bw.py` resizes the images and converts them to black and white. The commands to use to run them are specified at the top of each script. The images are stored in the subfolder `images`, and are in separate zip files in order to stay under the maximum file size in GitHub.
* **Train a colorization model** : we then trained a Convolutional Neural Network on the images in order to set up a model able to colorize greyscale images. The training is done in the folder `colorization_model`. The script `colorization_training.py` trains the CNN and saves the optimal parameters in `model1.pth`. `colorization_visu.py` allows to see the colorization on new images. 
*  **Develop app** : we developed the puzzle application and linked it to the trained model in order to colorize the puzzle images. The main code for the application is in `app.py`.

## Notes on running the scripts
The scripts relevant to the scraping and the training of the images don't have to run in order to run the app. Indeed, the model is already trained and does not need to be trained again. However, if you want to run these scripts, there are a few actions to take first:
* **Scraping** : a chrome webdriver is needed to run the script. To download it, see https://chromedriver.chromium.org/downloads.
* **Model training** : before running the script to train the model, unzip all color images (that can be found in `scraping_images/images`) into the subfolder `colorization_model/images_folder/color/color_images`.

## Example of a grayscale image colorized with the model



<p align="center">
  <img src="static/images/examples/_1.jpg" alt="Grayscale" width = "300">
  <br>
  <em>Original image</em>
</p>

<p align="center">
  <img src="static/images/examples_with_color/_1.jpg" alt="Colorized" width = "300">
  <br>
  <em>Colorized image</em>
</p>

</section>
<section id="track chosen :-dashboard-interactive-app" class="level3">
<h2 class="anchored" data-anchor-id="parcours-dashboard-application-interactive">Track chosen :dashboard-interactive-app</h2>
<div class="callout callout-style-default callout-tip callout-titled">
<div class="callout-header d-flex align-content-center">
<div class="callout-icon-container">
<i class="callout-icon"></i>
</div>
<div class="callout-title-container flex-fill">
<p><strong>Objective</strong> :</p>
</div>
</div>
<div class="callout-body-container callout-body">
<p>From an existing project or a new project, develop an interactive app or a <em>dashboard</em> addressing a business issue, then deploy it on a production infrastructure.</p>
</div>
</div>
<p><strong>Steps</strong> :</p>
<ul class="task-list">
<li><label><input type="checkbox">Fulfil the <em>checklist</em> of development best practices</label></li>
<li><label><input type="checkbox">Develop an interactive app <code>Streamlit</code> or a static <em>dashboard</em> with <code>Quarto</code> addressing a business issue</label></li>
<li><label><input type="checkbox">Create a <code>Docker</code> image for the local application</label></li>
<li><label><input type="checkbox">Deploy the app on <code>SSP Cloud</code> (interactive app) or on <code>Github Pages</code> (static website)</label></li>
<li><label><input type="checkbox">Customize the theme, the CSS etc. to best highlight the publication results and its main takeaway</label></li>
<li><label><input type="checkbox">Automate the ingestion of input data so that the <em>website</em> updates regularly</label></li>
<li><label><input type="checkbox">Industrialize deployment in <code>GitOps</code> mode with <code>ArgoCD</code></label></li>
<li><label><input type="checkbox">Monitor the app : <em>logs</em>, performance metrics, etc.</label></li>
</ul>


</div>
</div>
</section>
<section id="Checklist of development good practices" class="level2">
<h2 class="anchored" data-anchor-id="checklist-des-bonnes-pratiques-de-développement">Checklist of development good practices</h2>
<p>Below development good practices are a <strong>requirement of this course</strong>. They have to be applied to both <strong>group projects<strong> and </strong>individual code reviews</strong>.</p>
<ul class="task-list">
<li><label><input type="checkbox"><strong><code>Git</code></strong> usage </label>
<ul>
<li>Presence of a <code>.gitignore</code> file adapted to the language and with additional rules to respect good versioning practices</li>
<li>Collaborative work : use of branches and <em>pull requests</em></li>
</ul></li>
<li><label><input type="checkbox"><strong>Presence of a <code>README</code></strong> file introducing the project : context, objective, how to use it ?</label></li>
<li><label><input type="checkbox"><strong>Presence of a <code>LICENSE</code></strong> file stating the licence (<em>open-source</em>) for project exploitation.</label></li>
<li><label><input type="checkbox"><strong>Versioning of packages</strong> : presence of a <code>requirements.txt</code> file / environment file <code>environment.yml</code> for <code>conda</code></label></li>
<li><label><input type="checkbox"><strong>Code quality </strong></label>
<ul>
<li>Respect of community standards  : use a <em>linter</em> or a <em>formatter</em></li>
<li>Modularity : a main script calling modules</li>
</ul></li>
<li><label><input type="checkbox"><strong>Projects structure</strong></label>
<ul>
<li>Respect of community standards (<code>cookiecutter</code>)</li>
<li>Project modularity according to the model mentioned in the course:
<ul>
<li>Code on <code>GitHub</code></li>
<li>Data on <code>S3</code></li>
<li>Separated configuration files (<em>secrets</em>, etc.)</li>
</ul></li>
</ul></li>
</ul>

### Link to the CD git repo 
https://github.com/yann-Choho/puzzle-deployment.git
By using this link, you can access the git repository containing the development setup

### Link to the website of the application
https://puzzle.kub.sspcloud.fr/
By using this link, you can access directly the website application and enjoy it !
