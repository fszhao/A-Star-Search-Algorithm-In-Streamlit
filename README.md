# A* Search Algorithm Using Streamlit

## Overview:
 
Welcome to the A* Search Algorithm Visualization App! This Streamlit application demonstrates the A* search algorithm in action, allowing users to interactively explore how the algorithm finds the optimal path from a start to a goal in a grid-based environment.


## Setup

1. Install Anaconda for Python for your operating system: [Anaconda](https://www.anaconda.com/download/).
2. Execute:

    ```bash
    $ conda env create -f environment.yml
    $ conda activate astarstreamlit
    ```

3. Set up Jupyter notebook to use this environment:

    ```bash
    python -m ipykernel install --user --name astarstreamlit --display-name "Python (astarstreamlit)"
    ```

4. Open a terminal and run:

    ```bash
    $ streamlit run app.py
    ```

The app features a bunch of examples of what you can do with Streamlit. Jump to the [quickstart](#quickstart) section to understand how that all works.


## Features
- Map Choice: Select between two map sizes.

- Run A Algorithm:* Click the "Run A* Algorithm" button to visualize how the A* algorithm explores the grid, finding the shortest path from the start to the goal.


<img src="https://github.com/fszhao/A-Star-Search-Algorithm-In-Streamlit/blob/main/demo.gif" width=300 alt="Demo"></img>


## Resources

- https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html
- https://www.redblobgames.com/pathfinding/a-star/introduction.html 
