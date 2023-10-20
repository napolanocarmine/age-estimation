> ARTIFICIAL VISION
>
> **2020/2021**
>
> **CONTEST** **-** **AGE** **ESTIMATION**
>
> GENERAL DESCRIPTION OF THE PROBLEM
>
> • **Age** **estimation** from face images is nowadays a relevant
> feature in several real applications, such as digital signage, social
> robotics, business intelligence, access control and privacy
> preservation. In the era of deep learning, very effective DCNNs for
> age estimation has been proposed, reaching performance comparable to
> humans. However, the best methods proposed in the literature are based
> on ensembles of DCNNs, not easily applicable in applications requiring
> a real-time response (e.g. digital signage, social robotics). In
> addition, the training procedure is typically complex, since there is
> not yet a dataset that is at the same time large enough and reliably
> annotated for training standard DCNNs.
>
> ![](./brijsr5x.png){width="1.575in"
> height="0.793332239720035in"}• To this aim, we propose for this
> contest the usage of the biggest face dataset in the world,
> **VGGFace2**, which we have annotated with age labels. The age
> annotations are float numbers so that the participant can design the
> DCNN either as **regressors** or **classifiers**.Having available a
> wide dataset, the participants can define quick training procedures
> for standard DCNN architectures.Therefore, the validity of the proposed
> solutions can be evaluated by analyzing the **results** **on** **the**
> **test** **set** and the design choices in terms of
> **pre-processing**, **data**
> **augmentation**,**architecture**,**learning** **procedure** and
> **loss** **function**.
>
> DESCRIPTION OF THE DATASET
>
> • The **VGGFace2** dataset
> [[(https://github.com/ox-vgg/vgg_face2]{.underline})](https://github.com/ox-vgg/vgg_face2)
> is composed of more than 3 million images of 9.131 identities. It is
> already divided in **training** **set**
> [[(https://drive.google.com/file/d/1K56kVYHHDfLA2Anm7ga0tQolMwIPk6R8/view?usp=sharing]{.underline})](https://drive.google.com/file/d/1K56kVYHHDfLA2Anm7ga0tQolMwIPk6R8/view?usp=sharing),
> [including 8631 identities, and **test** **set**
> [(https://drive.google.com/file/d/1K47RjWJ-BYSMEw8V7U6FM2QUp3CO51M1/view?usp=sharing]{.underline}),including
> the remaining 500
> subjects.](https://drive.google.com/file/d/1K47RjWJ-BYSMEw8V7U6FM2QUp3CO51M1/view?usp=sharing)
>
> • Each image of the dataset depicts a **single** **face**;in spurious
> cases there are other small faces in the background, but they must be
> filtered out retaining only the **biggest** **face** in foreground.
>
> ![](./3f4wxtfo.png){width="1.575in"
> height="0.793332239720035in"}• We make available to the participants a
> CSV file with the **age** **labels** **of** **the** **training**
> **samples**
> [[(https://drive.google.com/file/d/1Jyv4nzltiaEBhjKiAvlD74tdzDFVeAA4/view?usp=sharing]{.underline})](https://drive.google.com/file/d/1Jyv4nzltiaEBhjKiAvlD74tdzDFVeAA4/view?usp=sharing).
> Each raw of the file includes, separated by a comma (according to the
> CSV standard),the relative path of the training sample (e.g.
> John_Doe/0000.jpg) and the age label as a float (e.g. 32.3452).
> Therefore, an example of raw may be John_Doe/0000.jpg,32.2452. If the
> participant designed the DCNN as a classifier,the age must be rounded
> to the nearest integer (e.g.21.3 is rounded to 21, while 21.6 is
> rounded to 22).
>
> ANNOTATION FORMAT
>
> • The annotations are very simple since they include only two
> fields,separated by comma: -- **Path** **of** **the** **image**:folder
> and filename
>
> ![](./k1ytszy3.png){width="1.575in"
> height="0.793332239720035in"}![](./bmyjwrqf.png){width="3.769998906386702in"
> height="4.02in"}-- **Age**: it is represented as a float value. In
> case you design your neural network as a classifier, use the value
> rounded to the nearest integer (e.g.21.6=22,21.3=21).
>
> DESCRIPTION OF THE CONTEST TASKS
>
> • The participants have to train their deep convolutional neural
> network (DCNN),apply it over all the samples of the test set and
> produce a report of the results according to these specifications:
>
> -- The filename must be **GROUP_ID.csv**,where GROUP_ID is the name of
> the group
>
> -- Each raw of the file must include, separated by a comma (according
> to the CSV standard), the relative path of the test sample (e.g.
> John_Doe/0000.jpg) and the estimated age as an integer (e.g. 32).
> Therefore, an example of raw may be John_Doe/0000.jpg,32,identical to
> the training annotation.
>
> -- If the participants designed the DCNN as a regressor and the age
> estimation is a float,the value must be rounded to the nearest integer
> (e.g.21.3 is rounded to 21,while 21.6 is rounded to 22).
>
> • In addition,all the participants must create a private Github
> repository in which they must include at least: -- **The** **trained**
> **DCNN**
>
> -- **The** **code** **used** **for** **training** **the** **DCNN** and
> detailed instructions for reproducing the experiment -- **The**
> **code** **for** **testing** **the** **DCNN** and detailed
> instructions for reproducing the experiment
>
> • Finally, the participants must prepare a **presentation** to
> describe their method, pointing out especially the choices done in
> terms of **pre-processing,data**
> **augmentation,architecture,learning** **procedure** **and** **loss**
> **function**.

