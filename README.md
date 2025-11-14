1.  Gathering training data
First, I went through the first 100 links in the list with a crawler. More than half of them turned out to be non-working. At first, I tried to extract data only from potentially significant parts of the pages (div, a, ...), but after checking the extracted information, I concluded that the most effective option was to extract information from the entire body or the entire page if the body was missing.
After checking the information that was extracted in the final version and discarding the pages that were extracted but still did not contain meaningful information, only 29 pages remained. I decided that this amount would be too small for training the model, so I decided to go through the first 200 links in the file; after repeating the entire procedure, 59 pages remained

2.  Tagging products
Using Label Studio, I tagged the products sold on all 59 pages (including products suggested below as "You might also like")

3.  Training new model
At first, I tried to train the "bert-base-cased" model, but I concluded that the data from 59 pages would be too little to train the model "from scratch". Tests showed that out of 704 sites (links to which were in the file), only 388 were working, and with the "bert-base-cased" model, only on 1% of the working sites was found any text.
After that, I tried the "dslim/bert-base-NER" model: results was the same.
After searching for pre-trained models on HuggingFace, I settled on "tner/roberta-large-ontonotes5" because it was trained on data that also contained products and, importantly, it was trained on English language data. This model already showed results that were significantly better.
After another pass through the list of sites, it was able to classify text on 71 pages, which is almost 18% of all working sites.
Also, after trying to increase the batch size during training from 8 to 16, the result improved, and now the model recognizes products on 109 sites out of 388, which is 28% of all working sites.

4.  Conclusions
I believe that at this stage, the model needs to be trained on more data or the site crawling needs to be optimized so that:
-   The data for annotation contains less "noise" (junk), making it easier for the model to find the pattern for locating PRODUCT on sites.
-   Cleaner data is fed into the model, making it easier to find results.

It is also worth considering the fact that although some sites provide a response, the extracted site does not contain the information we are looking for, or the domain has been bought by a company that is no longer in sales (while reviewing sites during labeling, I noticed that a couple of sites contain casino ads).