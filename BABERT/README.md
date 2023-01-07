<b>BA-BERT</b>
<p>BA-BERT is a BERT model that was fine-tuned on bio literature from pubmed.</p>

<p>The requirements.txt file lists all the necessary packages/libraries in order for the model to run.</p>
<b>Note: The .pth file containing the model's weights was too large to upload to GitHub.</b><br>
<br>
<b>Running BA-BERT</b>
<p>The model takes a string as input and outputs each token in the senence, the token's label, and the probability of that token actually being that label. Probabilities are sorted by id then ranked in descending order.</p>
<p>Example>
<p>Input: "The dog plasma was used to"</p>
<p>Output:</p>
<p>[['the', 1, 0.0291], ['dog', 1, 0.0273], ['plasma', 1, 0.0124], ['was', 3, 0.01232], ['used', 3, 0.01122], ['to', 4, 0.0732]]</p>
