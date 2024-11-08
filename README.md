# Advanced Topics in Machine Learning - RAG chatbot


## Running the `Project.ipynb` notebook

Before running the notebook, you should make sure that the `ipykernel` package is installed by running `pip install ipykernel`.

### Gemini free API key
You will need an API key to use Gemini. You can obtain a free API key from the Gemini website. You can find more information [here](https://ai.google.dev/pricing?authuser=1#1_5flash).

After obtaining an API key, you should copy the file `.env.template` into a new file called `.env` and replace `GOOGLE_API_KEY=...` by `GOOGLE_API_KEY=<your_api_key>`.

## Running the interface
You are encouraged to first run the notebook, since it already includes all the libraries needed to run the interface.

The interface is included in `.py` files, and can be run by executing the following command in the terminal:


```bash
chainlit run main.py
```

The browser should open automatically. If it does not, you can look at the terminal output to identify which port the application is using.


To continue developing the project, you can also run the following command to watch for changes in the code and automatically reload the interface:
```bash
chainlit run main.py -w
```

### Additional notes
Questions should always be posed in English. Additionally, note that the model can take a few seconds to answer a given question.

### Further information
You can check [chainlit.md](chainlit.md) for the README presented at the left of the interface.
