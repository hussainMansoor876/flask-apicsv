from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import pandas as pd
import csv
import nltk
nltk.download()


app = Flask(__name__)


CORS(app, allow_headers=["Content-Type", "Authorization",
                         "Access-Control-Allow-Credentials"], supports_credentials=True)



@app.route('/', methods=["POST"])
def index():
    fileData = request.files
    sent = pd.read_csv(fileData['csv'], sep="\t", quoting=csv.QUOTE_NONE)
    sent = str(sent.iloc[:, 0])
    def get_continuous_chunks(text, label):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        prev = None
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if type(subtree) == Tree and subtree.label() == label:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk
    data = get_continuous_chunks(sent, 'GPE')
    print(data)
    return jsonify({"places": data})


if __name__ == "__main__":
    app.run(debug=True)
