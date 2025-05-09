# System for classifying texts by literary genre based on a machine learning algorithm model

The For_diploma_script_collector_(1, 3 and 4) scripts parsed a collection of Russian literature works, distributed by genre in txt format (about 19320 works of 7 different genres: Роман, Повесть, Рассказ, Поэма, Пьеса, Статья, Очерк).
The books were taken from library websites - https://ilibrary.ru and https://lib.ru/.

Due to the hacker nature of the script (For_diploma_script_collector_1.py), there may be errors in the resulting material, use at your own risk!

After that, the collection was compressed for balance and scalability (books_max_balanced). Now it is about 14,000 works, about 1860-2050 works in each genre. All scripts for this are also included (For_diploma_collector_helper_2 and 3).

After that, 7 different classification algorithms were launched: NB, SVM, DT, FFBP, RNN, DAN2, CNN (For_diploma_CMs). Using various adjustments, the optimal accuracy values ​​were determined for each classification model (For_diploma_CM_1...7). The best models were those using the SVM classification algorithm.

Then a small software was written (literature_genre_classifier), with which you can determine the genre of a work in Russian in the form of text or a file of the format .txt, .pdf, .docx. 

Enjoy using it!
