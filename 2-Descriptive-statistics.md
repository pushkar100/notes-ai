<!-- TOC --><a name="descriptive-statistics-notes-udacity-class"></a>
# DESCRIPTIVE STATISTICS NOTES (UDACITY CLASS)

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [DESCRIPTIVE STATISTICS NOTES (UDACITY CLASS)](#descriptive-statistics-notes-udacity-class)
- [Lessons 1-3: Intro to Research Methods.](#lessons-1-3-intro-to-research-methods)
   * [Constructs:](#constructs)
   * [Variables:](#variables)
   * [Extraneous Factors (Lurking Variables):](#extraneous-factors-lurking-variables)
   * [Population & Sample:](#population-sample)
      + [Sampling Error:](#sampling-error)
      + [Parameter v/s Statistic:](#parameter-vs-statistic)
   * [Visualizing Relationships:](#visualizing-relationships)
   * [Correlation Does Not Prove Causation!:](#correlation-does-not-prove-causation)
      + [How to prove causation?](#how-to-prove-causation)
   * [Surveys v/s Controlled Experiments:](#surveys-vs-controlled-experiments)
      + [Surveys: ](#surveys)
      + [Controlled experiments: ](#controlled-experiments)
   * [Random Assigment:](#random-assigment)
      + [Random v/s Convenience Samples:](#random-vs-convenience-samples)
- [Lessons 4-6: Data Visualization](#lessons-4-6-data-visualization)
   * [Frequency chart: ](#frequency-chart)
      + [Frequency Count](#frequency-count)
      + [Relative Frequency:](#relative-frequency)
      + [Histograms:](#histograms)
         - [Uniform Distribution:](#uniform-distribution)
         - [Normal Distribution:](#normal-distribution)
         - [Skewed Distribution:](#skewed-distribution)
- [Lesson 7: Working with Google Spreadsheets](#lesson-7-working-with-google-spreadsheets)
- [Lessons 8-10: Central Tendency](#lessons-8-10-central-tendency)
   * [Choosing a number to represent typical data:](#choosing-a-number-to-represent-typical-data)
      + [Modes:](#modes)
         - [No modes:](#no-modes)
         - [Multiple modes:](#multiple-modes)
      + [Mean:](#mean)
         - [Outliers:](#outliers)
      + [Median:](#median)
         - [Median for even number of values:](#median-for-even-number-of-values)
      + [Mean, Median and Mode in Normal Distribution:](#mean-median-and-mode-in-normal-distribution)
      + [Mean, Median and Mode in Skewed Distribution:](#mean-median-and-mode-in-skewed-distribution)
- [Lessons 11-12: Variability](#lessons-11-12-variability)
   * [Range of a graph:](#range-of-a-graph)
      + [Chopping off the tails:](#chopping-off-the-tails)
         - [Calculating the Quartiles:](#calculating-the-quartiles)
         - [Mathematically Defining Outliers using IQR:](#mathematically-defining-outliers-using-iqr)
         - [Mean and the IQR:](#mean-and-the-iqr)
         - [Boxplots:](#boxplots)
         - [Disadvantage of IQR:](#disadvantage-of-iqr)
      + [Better Ways to Calculate Spread:](#better-ways-to-calculate-spread)
         - [Deviation:](#deviation)
         - [Disadvantage of Average Deviation:](#disadvantage-of-average-deviation)
         - [Squared Deviations:](#squared-deviations)
         - [Standard Deviation (σ):](#standard-deviation-)
         - [Why use Standard Deviation(σ)? ](#why-use-standard-deviation)
         - [Bessel's Correction:](#bessels-correction)
- [Lessons 13-15: Standardizing](#lessons-13-15-standardizing)
         - [Finding certain proportion of the sample that falls between two bin values:](#finding-certain-proportion-of-the-sample-that-falls-between-two-bin-values)
         - [Disadvantage of Relative Frequency:](#disadvantage-of-relative-frequency)
      + [Continuous Distribution:](#continuous-distribution)
      + [Standardized Score:](#standardized-score)
         - [Z-score:](#z-score)
            * [Mean(x̅) in terms of Z-score:](#meanx-in-terms-of-z-score)
         - [Standard Deviation(σ) in terms of Z-score:](#standard-deviation-in-terms-of-z-score)
      + [Standard Normal Distribution:](#standard-normal-distribution)
- [Lessons 16-17: Normal Distribution](#lessons-16-17-normal-distribution)
         - [Z-table:](#z-table)
- [Lessons 18-19: Sampling Distributions](#lessons-18-19-sampling-distributions)
      + [Mean of the sample means(M):](#mean-of-the-sample-meansm)
         - [Relationship between `σ` & `SE`:](#relationship-between-se)
      + [Central Limit Theorem:](#central-limit-theorem)
         - [Central Limit Theorem Revisited:](#central-limit-theorem-revisited)

<!-- TOC end -->

Prerequisite 1 for taking up the Machine Learning Course on Udacity.

<!-- TOC --><a name="lessons-1-3-intro-to-research-methods"></a>
# Lessons 1-3: Intro to Research Methods.

The 3 Factors that are always important to conduct a good research:
1. The size of the sample set (bigger the better)
2. The focus of the sample set (Ex: Asking *students* about exams)
3. The methodology used for surveying/conducting experiment (sound methodlogy)

<!-- TOC --><a name="constructs"></a>
## Constructs:

There are certain things for which there cannot exist just one method to correctly measure them. 
Examples of such things include happiness, guilt, memory, etc and they are known as **Constructs** (Very difficult to define and measure).

Using **Operational Definitions**, we can measure Constructs in the real-world. What is Operationa Definition?:
http://www.pqsystems.com/qualityadvisor/DataCollectionTools/operational_definition.php
(Operational definition removes ambiguity while measuring something that could be interpreted differently by different people.)

Operational definition:
- Is a way of turning constructs into variables we can measure, and
- A way of describing a variable in terms of the way we measure it.

Tip: 
- *Gallons of water* is NOT a construct since it can be easily measured in *gallons*. 
- *Age* is a construct since different people might consider age differently, like by DOB, Maturity, Height, etc.
- *Age by DOB* is NOT a construct since we can measure it by DOB.
- *Annual Salary* is a construct since different people might consider it differently, like by USDs, Yens, Property, etc.
- *Annual Salary in USD* is NOT a construct  since we can measure it in *USDs*.

<!-- TOC --><a name="variables"></a>
## Variables:

A variable is a value that may change or differ between individuals in an experiment. Scores on an intelligence test qualify, since each person may have a different score.

**Note:** Statements about the relationships between variables are called **Hypotheses**. (Ex: 'Students who *sleep* more before an exam tend to get better *marks* in the exam').

<!-- TOC --><a name="extraneous-factors-lurking-variables"></a>
## Extraneous Factors (Lurking Variables):

**Extraneous Factors** are those things that can influence Constructs. (Ex: Sleep factor before taking a memory test).
If a certain factor is made constant (Like conducting an experiment at the same time when time is considered as a factor), 
it is better for the outcome/conclusion. (Ex: If everyone takes a memory test at the same time [Acc. to their respective timezones] then the results can be trusted better.)

The more extraneous factors that are constant, the easier it is to trust the conclusion we arrive at from analysing the data. 
The reason for this is that these lurking variables usually ambush our results (if they keep varying between each item) (lurking to attack analogy).

<!-- TOC --><a name="population-sample"></a>
## Population & Sample:

(Refer: https://classroom.udacity.com/courses/ud827/lessons/1293178557/concepts/565334210923)

Population is the entire set of items included in the experiment and the Sample is a subset of the population. 
The average result of the experiment for the population is known as `μ (myu)`. [Population average]
The average result of the experiment for the sample set is known as `x̅ (x bar)`. [Sample average]

**Note**: A population is a group of phenomena that have something in common. The term often refers to a group of people, as in the following examples:
- All registered voters in Crawford County
- All members of the International Machinists Union
- All Americans who played golf at least once in the past year

But populations can refer to things as well as people:
- All widgets produced last Tuesday by the Acme Widget Company
- All daily maximum temperatures in July for major U.S. cities
- All basal ganglia cells from a particular rhesus monkey

The average of the population `μ (myu)` could be the same as the average of the sample `x̅ (x bar)` but the opposite is seldom true since there 
would exist other samples within the population, some of which are mutually exclusive to the considered sample.
(So their average will not be the same).

Therefore, *generally* μ (myu) and x̅ (x bar) won't be exactly the same. Also, a *good* sample set is important to conclude a research.

A bigger sample will better approximate the population. Therefore, bigger samples are more useful.

<!-- TOC --><a name="sampling-error"></a>
### Sampling Error:

The difference between μ (myu) & x̅ (x bar) is called *Sampling Error* `μ - x̅`.

<!-- TOC --><a name="parameter-vs-statistic"></a>
### Parameter v/s Statistic:

A parameter is a characteristic of a population, while a statistic is a characteristic of a sample.
Therefore, μ (myu) is a **parameter** and x̅ (x bar) is a **statistic**.

Also, the **number** of people/items included in the **sample** is denoted by `n` and going by the above definitions, `n` 
is also a **statistic**.

Refer: https://www.cliffsnotes.com/study-guides/statistics/sampling/populations-samples-parameters-and-statistics

<!-- TOC --><a name="visualizing-relationships"></a>
## Visualizing Relationships:

When plotting a graph between data, we use the `x axis` to represent the *Independent Variable/Predictor Variable* and 
on the `y axis` we plot the *Dependent Variable/Outcome*.

An example of this is plotting 'Hours of sleep' on the x-axis (independent) and 'Memory Test Score' on the y-axis (dependent).

<!-- TOC --><a name="correlation-does-not-prove-causation"></a>
## Correlation Does Not Prove Causation!:

What this means is that even though there is a relationship between independent & outcome, it does NOT prove that 
the independent is the CAUSE for the outcome.

Example: If the relationship between sleep and memory score suggests that more sleep sees better memory scores, it does not mean that 
a person who has slept more will always score more (Someone who has slept less may still score the same). Therefore, the correlation 
CANNOT be cause for causation.

The reasons for this could be the existence of OTHER extraneous factors (Other lurking variables).

A fun example that shows correlation cannot prove causation is the **Golden Arches Theory**. 
(https://mediawiki.middlebury.edu/wiki/IPE/Golden_Arches_Theory_of_Conflict_Prevention) 
(Answer: Opening a McDonald's in other countries *cannot* be the cause for that country to not go to war with the USA).

**Note:** If there is a correlation, then we can *predict* something but *not prove it*.

<!-- TOC --><a name="how-to-prove-causation"></a>
### How to prove causation?

For showing a relationship/correlation, we need to undertake observational studies/surveys (like asking people as part of a survey).
In order to show causation, we need to perform a **Controlled Experiment**.

<!-- TOC --><a name="surveys-vs-controlled-experiments"></a>
## Surveys v/s Controlled Experiments:

<!-- TOC --><a name="surveys"></a>
### Surveys: 

Advantages of surveys:
1. Surveys are inexpensive.
2. Easy way to get info on a population.
3. Can be done remotely. (via mail, for example.)
4. Anyone can access and analyze survey results. (They are timeless)

Disadvantages of surveys:
1. Biased responses.
2. Respondents not understanding the question. (1 & 2 can be called **Response bias** and is a negative aspect of a survey)
3. Untruthful responses.
4. Respondents refusing to answer. (It is known as **Non-response bias** and is a negative aspect of a survey)

Surveys are used to probe *Constructs* and these constructs can have a lot of definitions. Surveys need to contain carefully worded questions.

<!-- TOC --><a name="controlled-experiments"></a>
### Controlled experiments: 

The definition of a controlled experiment is a test where the person conducting the test only changes one variable at a time 
in order to isolate the results. An experiment where all subjects involved in the experiment are treated exactly the same 
except for one deviation is an example of a control experiment.

Basically, in controlled experiments, we control the variables/factors.

Example: Placebo: Give a random set of people a sleep pill (half the people get an 'active' pill and half get an 'inactive' pill). The 
experiment is to test whether the 'active' pill has an effect on the user's sleep. The participants are NOT told if they are receiving an active
pill or an inactive pill. The researchers want all the people to believe that they are taking medication(active pill). 
This process helps eliminate bias from the outcome. [The inactive pill is a **placebo** and the case where people who take the placebo and felt better 
is known as the **placebo effect**. If people know they are taking the placebo then they might subconsciously think that 
they are not doing anything and hence it will influence the results.]

Basically, placebos can help identify lurking variables(extraneous factors).

Note:
- Single Blind: Only the participants don't know which pill they are taking. 
- Double Blind: Even the researchers measuring the participants do not know which pills they have taken. 
(Double blind increases the accuracy of the experiment)

Extraneous factors need to be controlled in order to get causal results.

**Note:** In a controlled experiment, the researcher seeks to:
- Manipulate the independent variable.
- Measure the changes in the dependent variable or outcome.
- Control the extraneous/lurking variable.

<!-- TOC --><a name="random-assigment"></a>
## Random Assigment:

Random assignments could help **minimize** some of the lurking variables/extraneous factors. 

Example: Say an experiment needed to be done 
to measure if a pill had a certain effect on people. If a working pill was given to men alone and a non-working pill to women alone and 
the result showed men sleeping more, it could mean two things, either the pill worked or that women in general sleep more. This result would be 
inconclusive.

If the pills had been given out randomly, then both men and women would be a part of both the pill groups. This would lead to a better conclusion.

In a Random assignment, we: 
- Select individuals in such a way that everyone has the same chance of being selected.
- Select individuals in such a way that selection of one individual has no effect on anyone else's chances of being selected.

Random assignment works better on *larger samples* (closer to the population).

Examples where random assignment works well are Gender, Age, Wealth, etc (depending upon the context of the experiment to be conducted).

<!-- TOC --><a name="random-vs-convenience-samples"></a>
### Random v/s Convenience Samples:

Random samples are assumed to NOT be biased and treats every perso equally (like mentioned above). Convenience samples are the opposite of random samples since it forms the set of people who are conveniently picked (Example: Friends & relatives). They may be biased and not be representative of the population.

<!-- TOC --><a name="lessons-4-6-data-visualization"></a>
# Lessons 4-6: Data Visualization

<!-- TOC --><a name="frequency-chart"></a>
## Frequency chart: 
A frequency table is a table that shows the total for each category or group of data. 
**OR**
A Frequency Table is a table that lists items and uses tally marks to record and show the number of times they occur.

<!-- TOC --><a name="frequency-count"></a>
### Frequency Count
A **frequency count** is a measure of the number of times that an event occurs.

<!-- TOC --><a name="relative-frequency"></a>
### Relative Frequency:
The **relative frequency** of an event is defined as the number of times that the event occurs during experimental trials, divided by the total number of trials conducted.

To compute relative frequency, one obtains a frequency count for the total population and a frequency count for a subgroup of the population. The relative frequency for the subgroup is:

`Relative frequency = Subgroup count / Total count`

The above equation expresses relative frequency as a **proportion**. It is also often expressed as a **percentage**. Thus, a relative frequency of 0.50 (proportion) is equivalent to a percentage of 50%.

`percentage = proportion * 100`

**Note:**
- **All proportion are greater than or equal to `0` and less than or equal to `1`**
- **The sum of the proportions of all the items in the frequency table must be equal to `1`**
- **The sum of the percentages of all items in the frequency table must be equal to `100%`**

It is important how we group the data into a frequency chart - and it depends on how we want to view the data. (Example: If we want to see how many students watch Udacity videos from each country, we group the data country-wise on the frequency table. Similarly, if we want to view how many students view the videos from each continent, we'd group the data continent-wise.)

<!-- TOC --><a name="histograms"></a>
### Histograms:
We can build **histograms** (Similar to Bar graphs but divided by **ranges** on the `x-axis` and by the **frequency** on the `y-axis`) easily from a given frequency table/chart. 

Bar graphs have distinct values or categories on the x-axis (No ranges) and their order of appearance does not really matter whereas in a histogram the values are ranges (we can change the ranges to depict smaller or bigger groups of x-axis values) and their order of appearance does matter - *Only one order* (Ex:  11-20 interval should come before 21-30 interval).

- Histogram x-axis variable: Numerical & Quantitative (Ex: 0 to 100)
- Bar graph x-axis variable: Categorical or Qualitative (Ex: Asia, Europe, Africa, etc)

(Refer: https://www.mathsisfun.com/data/histograms.html)

The length of the ranges on the x-axis are known as **bin sizes** or **interval sizes**. The bin size should be optimal, not too small nor too big since it could become difficult to visualize the data. The bin/interval size must be the **same** for all the bins.

(Interactive Histogram: http://www.shodor.org/interactivate/activities/Histogram/)

Sometimes, in a histogram, we compromise detail for convenience. For example, trying to get the exact number of students whose age is less than 20 when the interval contains 20 (20 is within the bin and not at the boundary of the bin) inside it is not possible (although we can get a rough estimate).

- With frequency tables, we have exact counts, so we can always create the histogram. But not the opposite way around.
- With tables, you can add the frequencies in each bin. With a histogram, you don't always know the exact frequencies (Compromise detail for convenience).
- A histogram lets you visually see the shape of a distribution.

**Note**: When we make a bin size larger, more values will fall into that bin and hence, the frequency for that bin increases.

<!-- TOC --><a name="uniform-distribution"></a>
#### Uniform Distribution:

The uniform distribution gets its name from the fact that the probabilities for all outcomes are the same. Every outcome is equally likely to occur. 

The histogram for a uniform distribution will look like a *rectangle*. The frequencies of all the bins/intervals will be the *same (equal)*. For roughly uniform distributions, the frequencies will be very similar (in case they aren't the same).

A deck of cards has a uniform distribution because the likelihood of drawing a heart, a club, a diamond or a spade is equally likely. A coin also has a uniform distribution because the probability of getting either heads or tails in a coin toss is the same.

(Refer: http://study.com/academy/lesson/uniform-distribution-in-statistics-definition-examples.html)

<!-- TOC --><a name="normal-distribution"></a>
#### Normal Distribution:
**Normal Distribution** is a function that represents the distribution of many random variables as a symmetrical bell-shaped graph.

Even on a histogram, the normal distribution would appear like a bell-shaped graph. It would be roughly symmetrical, tapering off at the extremes.

(Refer: http://whatis.techtarget.com/definition/normal-distribution)

<!-- TOC --><a name="skewed-distribution"></a>
#### Skewed Distribution:
Skewed distribution is the opposite of a normal distribution.

**Positively Skewed Distribution** on a graph means the frequencies are more on the **left side** than on the *right*. (Also known as **right-skewed** distribution)

**Negatively Skewed Distribution** on a graph means the frequencies are more on the **right side** than on the *left*. (Also known as **left-skewed** distribution)

**Note:** The "skew" does **not** refer to the peak but instead to the tapered side (known as the **tail**). Normal distribution has two tails, one on either side of the peak. In right/positive skew, the tail is more on the positive/right side of x-axis (greater values' side). In the left/negative skew, the tail is more on the negative/left side of the x-axis (lower values' side).

(Refer: http://www.statisticshowto.com/skewed-distribution/)

<!-- TOC --><a name="lesson-7-working-with-google-spreadsheets"></a>
# Lesson 7: Working with Google Spreadsheets

Refer: https://classroom.udacity.com/courses/ud827/lessons/116708348/concepts/1164749820923

**Formulae: (To help you with statistics and google sheets):**
- **Mean** or sample mean is another term for sample average. Therefore, `mean = x̅ (x bar)`. (Sum of all the observed outcomes  from the sample divided by the total number of events)
- **Deviation** of a particular result is the difference between one particular value and the mean(x̅).
- **Squared deviation** of a particular result is the square of its deviation from the mean(x̅).
- **Variance** is the *Average* of the *Sum* of the *Squared deviations* divided by `n` (n = the count of the sample set). (In other words, variance is the average of the squared differences from the Mean.)
- **Standard deviation** is the *Square root* of the *variance*. The Standard Deviation is a measure of how spread out numbers are. Standard deviation is denoted by `σ (sigma)`.

**Bonus:**
- **Median** is the "middle" of a sorted list of numbers. To find the Median, place the numbers in value order and find the middle. 
- **Median in the case of even set of numbers:** With an even amount of numbers things are slightly different. In that case we find the middle pair of numbers, and then find the value that is half way between them. This is easily done by adding them together and dividing by two.

**Google sheets functions:**
- average()
- sum()
- sqrt()

<!-- TOC --><a name="lessons-8-10-central-tendency"></a>
# Lessons 8-10: Central Tendency

<!-- TOC --><a name="choosing-a-number-to-represent-typical-data"></a>
## Choosing a number to represent typical data:

There maybe more than one number used to represent typical data. Example: What is the typical salary of a post-graduate engineer? Often, we can use different types of numbers to answer such questions.

1. **Mode** is the value at which the *frequency is the highest*. **(OR)** The *most common value* is the called the mode. (The number that *occurs the most*)
	- On a histogram, this is the bin or interval for which the frequency is the *Highest*.  The mode occurs on the `x-axis`.
	- For a frequency table, it is the value with the highest frequency (count).
2. **Median** is the value in the middle of the distribution (already defined earlier - check).
3. **Average/Mean** is that specific spot which rests in the center of the distribution, which is the average of all the values (already defined earlier - check).

<!-- TOC --><a name="modes"></a>
### Modes:
<!-- TOC --><a name="no-modes"></a>
#### No modes:
Some distributions have no modes. Example: **Uniform Distribution**. Since all the bins' frequencies are the *same* there is not one value which occurs the most. Hence, no mode.

<!-- TOC --><a name="multiple-modes"></a>
#### Multiple modes:
Some distributions have multiple modes.  These are called **Bi-modal Distributions**.

Example: Shoe sizes histogram - women mostly have size 7 and men mostly have size 9 (Therefore, we can see two humps on the graph, even though there is only one value with the highest frequency).

(Refer: http://www.statisticshowto.com/what-is-a-bimodal-distribution/)

**Note: Global Maxima v/s Local Maxima:**
- When we take the entire sample set and find one mode or peak (most commonly occurring value) then that value is known as the **Global maxima**.
- If we split the graph into regions (like in the case of shoe sizes for men & women example) and find the peak/mode for those regions separately then those values become the **Local maxima** for that respective region.

**More on modes:**
- The mode can describe any type of data - *Categorical(qualitative)* as well as *Quantitative(numerical)*.
- All scores in the dataset **DON'T** affect the mode. (For example: if mode value (say x)  has frequency 100 and another value (say y) exists with frequency 10. If we add another score to y, increasing it's frequency to 11, the mode value (frequency of x) does not change since it is still the highest. (Therefore, all scores do not affect the mode and it holds true for adding, deleting or modifying a score). - see outliers.
- There is **NO** equation for the mode.
- The mode is *likely to change* from one sample set to another. The mode *need not be same in all the samples*.
- The mode changes with the *bin/interval size* on a histogram: If the bin is made large enough, most values will fit in there and it will have the highest frequency, so the mode will be that bin. If made smaller, the mode might fit into some other bin instead of the current one. 

The **mode** is **not affected by outliers** at all! (see outliers under 'mean' section)

**Note:** Find mode on google spreadsheets: Use the `=mode(<range>)` function (Example: `=mode(A2:A28)` and hit <enter>.
(Or refer to this video: https://www.youtube.com/watch?v=TC0ZGYEf8dU)

<!-- TOC --><a name="mean"></a>
### Mean:
Mean is the average of all the *sample* data.

`Mean = sum of all n sample data / n`. 

It has an equation and hence it is used alongside *mode* for data representation. **Mean/average** is represented by `x̅ (x bar)`

**Mathematical Equations for Mean:**

For a **Sample set**:
`x̅ = Σx / n`  
where `Σx` is the sum of all the values in the sample data set (`x1+x2+..+xn`) and `n` is the number of values in the sample data set. (Sample average)

For an entire **Population**,:
`x̅ = Σx / N`  
where `Σx` is the sum of all the values in the entire population (`x1+x2+..+xN`) and `N` is the number of all values in the entire population. (Population average)

**Notes on mean:**
- *All* the scores in the distribution *affect* the mean. (That is, the mean will change if we add an extreme value to the dataset. For example, adding 10000 to a dataset of 10 elements with a mean of 7 will change the mean drastically).
- The mean can be described by a **formula** (unlike mode).
- **MOST IMPORTANTLY: Many samples from the same population will have similar means** (unlike mode). (Explained in detail later - but this can be seen to be true with random samples).

<!-- TOC --><a name="outliers"></a>
#### Outliers:
**Outliers** are those values in data that can be *unexpectedly different* from the rest of the values.

For example, in the set 10, 96, 100, 5, 12, 67, 100000, 50 the value 100000 is the outliner because it is an extreme value w.r.t the other ones in the set. 

When outliers exist, the **mean will be greatly affected** and the mean will **not be an accurate representation of the data (misleading).** It makes the mean a lot less representative of the middle of the data.

A **median** might be more representative of the middle data. It is less affected by the outliers compared to the mean.

Note that the **mode** is **not affected by outliers** at all!

<!-- TOC --><a name="median"></a>
### Median:
The median is the middle value of a data set. But, it is not just any middle value - it is the middle value when the data set is **sorted into order**. From *least-to-greatest* or *greatest-to-least* - both orders are okay!

Example: Find median of 5, 3, 7, 10, 6?
Answer:  Sort them as 3, 5, 6, 7, 10. The middle value is 6. Median = `6`.

<!-- TOC --><a name="median-for-even-number-of-values"></a>
#### Median for even number of values:
When odd number of values exist, it is easy to pick the middle one after sorting them. But, for even set of values, there are two middle numbers (instead of just one). In this case, we find the middle point of those two middle numbers. That is, median will be the **average of the middle two values**.

Example: Find median of 5, 3, 7, 10, 4, 6?
Answer:  Sort them as 3, 4, 5, 6, 7, 10. The middle values are 5 & 6. Median = (5 + 6) / 2 = `5.5`.

**Note**:  Median is known to have a **Robust** tendency. What this means is that it is not affected much by departures from the norm (caused by outliers). Hence, it is more reliable than the mean to calculate the middle point of the distribution when data contains outliers. (Only slightly affected by outliers).

The median is generally used as a measure whenever we deal with *highly skewed distributions*.

**Formula for median:**
For an ordered(sorted) dataset `x1, x2, x3, .. , x(n-1), xn` the median is defined as:
- For odd elements: `Median = x(n+1/2)`, and
- For even elements: `Median = ( x(n/2) + x(n+1/2) ) / 2`.

**Finding the median on a histogram**:
Sum up all the frequencies. Find the median number (depending on odd or even ~= half the sum of frequencies). The value (of the bin) at the median number is the median.

<!-- TOC --><a name="mean-median-and-mode-in-normal-distribution"></a>
### Mean, Median and Mode in Normal Distribution:
In a normal distribution, since the peak is in the middle the mode will be in the middle (highest frequency bin). Also, since the distribution is symmetric, the median and the mean will also be in the middle.

Therefore, `mean(x̅) = median = mode` for **Normal Distribution**.

<!-- TOC --><a name="mean-median-and-mode-in-skewed-distribution"></a>
### Mean, Median and Mode in Skewed Distribution:
In a Skewed distribution, the bin with the highest frequency will be to one side of the distribution. That bin represents the mode. 

The mean and median will be affected by the tail (outliers). But, the median being more robust, will not move towards the tail side as much as the mean does. 

Therefore:
- `mean(x̅) < median < mode` for **Negative Skewed Distribution**.
- `mode > median > mean(x̅)` for **Positive Skewed Distribution**.
(Sometimes they could be `<=` or `>=` also)

<!-- TOC --><a name="lessons-11-12-variability"></a>
# Lessons 11-12: Variability

Some graphs have almost same mean, mode and median. How do we differentiate between such graphs? 

Imagine two normally distributed graphs with same mean/mode/median. One graph is spread out more while the other hasn't spread out much. We can use this spread as a variable to measure the difference between these two graphs.

<!-- TOC --><a name="range-of-a-graph"></a>
## Range of a graph:
Range of a graph is defined as the difference between its two most extreme values (`highest value - lowest value`) on the *x-axis* (Y-axis generally being the frequency). 
- If the range is **BIG** then it means that the graph is more **spread out**.
- If the range is **SMALL** then it means that the graph is more **consistent**.

**Range has a drawback:** If new values are added to the graph then the range might sometimes change. If the new value falls within the existing range then the range remains same. If the new value falls outside the existing range, the range cannot be considered constant.

Therefore, the **variability** of a range is affected by outliers.

<!-- TOC --><a name="chopping-off-the-tails"></a>
### Chopping off the tails:
Statisticians generally want to avoid the outliers from affecting the range and their calculations. Therefore, we chop off the top 25% and bottom 25% of the graph from our calculations.

Each of the 25% range is known as a `Quartile(Q)`. We discard the first quartile `Q1` (bottom) and `Q4` (top).

The outliers now cannot affect our calculations since our focus will be on `Q2(25%-50%)` & `Q3(50%-75%)`

<!-- TOC --><a name="calculating-the-quartiles"></a>
#### Calculating the Quartiles:
We need to use **median** to calculate the quartile ranges. Calculate the median number of the whole graph (by adding frequencies if the data representation is a histogram). 

If `A` is the sum of frequencies then value at `A/2` is the median. The first quartile(`Q1`) will be the value present at half the median `(A/4)`. The second quartile(`Q2`) is the value at the median(`A/2`). The third quartile(`Q3`) will be the value present at three quarters of the frequency `(3A/4)`.

Our **new range** (after chopping off the tails) will now be the space **between `Q1` and `Q3`**.

`Q1` is known as the **Lower Quartile**.
`Q3` is known as the **Upper Quartile**.

(Refer: https://classroom.udacity.com/courses/ud827/lessons/1471748603/concepts/838774410923)

The difference between Q1 & Q3 is known as **Interquartile Range(IQR)**. `IQR = Q3 - Q1`

**Note:**
- About 50% of data falls within the IQR.
- It is **not** affected by every value in the dataset. (If the value within the boundary of the IQR changes to a value greater than or less than the IQR range then yes, it is affected, else it is not)
- It is **not** affected by outliers. Changing the value of outliers does not alter Q1 & Q2.

<!-- TOC --><a name="mathematically-defining-outliers-using-iqr"></a>
#### Mathematically Defining Outliers using IQR:
Not everyone value outside of the IQR can be termed as an outlier. For example, if the range of IQR is from 1000 to 5000 then the values such a 890 and 5600 are not that unexpected or far from the range. Therefore, we need to statistically define what exactly is an outlier.

**Definition:**
For a value `p` to be considered as an **outlier**, it should obey *either one* of the following inequalities:
`p < (Q1 - 1.5*IQR)` **or** `p > (Q3 + 1.5*IQR)`

<!-- TOC --><a name="mean-and-the-iqr"></a>
#### Mean and the IQR:
Generally, the mean does fall within the IQR in most cases. The cases for which the mean **may not** fall under the IQR could be when there a *presence of outliers*. Mean is very sensitive to outliers while the IQR is not affected by them. Hence, in this case the mean might fall outside of the IQR.

<!-- TOC --><a name="boxplots"></a>
#### Boxplots:
IQR graphs are visualized using a **Boxplot**. It contains a line with a box in the center representing the IQR (`Q1` to `Q3`) with a *line cutting across the box* representing `Q2` (the **median**). The outer line extends in either direction to represent the *minimum* and *maximum* values (Outside of which lie the outliers). Values are represented by **dots** and the *outliers* are represented by **dots outside from the min or max** depending on the value.

(Watch this video: https://www.youtube.com/watch?v=CoVf1jLxgj4)

<!-- TOC --><a name="disadvantage-of-iqr"></a>
#### Disadvantage of IQR:
We used or defined IQR to get rid of the problem caused by outliers to the range. The **range** was initially calculated to measure the **difference in spread** between two graphs with almost same mean, mode and median.

In the IQR, we can *identify the difference in spread without worrying about outliers giving unexpected results*. However, we still cannot determine the type of graph distribution.

There maybe graphs with **same IQR** but **different distributions**. Example: One graph with normal distribution, other with bimodal, and another with uniform. Using just IQR, we cannot differentiate between these types of graphs.

<!-- TOC --><a name="better-ways-to-calculate-spread"></a>
### Better Ways to Calculate Spread:
Range and IQR both have issues while trying to measure spread. The have too much **variability** or they cannot distinguish between types of graph based on distribution types.

<!-- TOC --><a name="deviation"></a>
#### Deviation:
Measuring the deviation from the data value to the **mean** value could provide a way to determine how far apart are the values placed.

Deviation = difference between mean/average (x̅) and the data value for a particular point. 

For a dataset comprising of values: `x1, x2, x3, ..., x(n-1), xn`, the deviation for one of the values (say x(i)) can be defined as: 
`Deviation = x̅ - x(i)` **OR** `Deviation = x(i) - x̅` (Choose any one and stick to it).

**Average Deviation:**
Average deviation is the average of the deviations of all the dataset values.
`Average Deviation = Σ(x̅ - x(i)) / n` where `n` is the size of the sample dataset.

<!-- TOC --><a name="disadvantage-of-average-deviation"></a>
#### Disadvantage of Average Deviation:
We came up with deviation & average deviation to measure *spread* of distribution (better than range & IQR).

The issue with deviations are that there could be negative values of a deviation based on its value w.r.t the **mean**. Therefore, when you take the **Average deviation** you may end up getting a `0` result (caused because the negative values nullify the positive values). Such a result cannot give us the true deviation/spread.

**Solution:**
1. Use **Absolute deviations**: `x(i) ~ x̅` or `|x(i) - x̅|`. Then, **Average absolute deviation** would be `Σ(|x(i) - x̅|) / 2`.
2. Use **Squared deviations** since squaring would remove the negative sign as well. *(This metric is used commonly)*

<!-- TOC --><a name="squared-deviations"></a>
#### Squared Deviations:
The formulae for all calculations related to squared deviations are:
1. A `Squared deviation = (x(i) - x̅)^2` (**Alternately, we can also use** `(x̅ - x(i))^2`. Choose one formula and stick to it)
2. **Sum of Squared Deviations(SS)** is calculated as follows:  `SS = Σ( (x(i) - x̅)^2 )`
3. **Average Squared Deviation** is calculated as `SS / n` where `n` is the sample dataset size.

The *Average Squared Deviation* has a special term known as **Variance**. Hence, `Variance = SS / n`.

**Note:** "Variance is the average of all the squared deviations"

**Important:** It is necessary to visualize Variance to understand it. Watch one of the course videos to help you better grasp the concept of variance: https://classroom.udacity.com/courses/ud827/lessons/1471748603/concepts/839344310923)

<!-- TOC --><a name="standard-deviation-"></a>
#### Standard Deviation (σ):
Once we have variance, we need to find out its square root to get back the deviation (Since we had squared the deviations in the first place - in order to prevent opposite sign deviations from canceling each other and proving useless). This is the standard method of find out the *variability* or *spread* and it is known as the **Standard Deviation** and is denoted by `σ`.

Standard deviation formula: `σ = √(Variance)` (i.e Square root of variance)

<!-- TOC --><a name="why-use-standard-deviation"></a>
#### Why use Standard Deviation(σ)? 
We can use absolute deviations an find the average. But, standard deviation has a special property that helps us more than absolute deviations.

If we take a **normal distribution** (bell-curve) where `mean = mode = median` and is symmetrical then:
- **65%** of the data falls between *1 standard deviation of the mean*.
- **95%** of the data falls between *2 standard deviations of the mean*.

Also:
- **99%** of the data falls between *3 standard deviations of the mean*.

(Visualization: https://www.youtube.com/watch?v=MoW3hMq-eIc and https://www.youtube.com/watch?v=MRqtXL2WX2M&t=62s)

**Math:**
- `1 standard deviation = mean(x̅) +/- 1σ`
- `2 standard deviation = mean(x̅) +/- 2σ` .. and so on.

There is a table that exists for telling how much percentage of data falls within what standard deviations of the mean.

**Note:** Don't forget that standard deviation is a **measure of spread**.

<!-- TOC --><a name="bessels-correction"></a>
#### Bessel's Correction:
Generally when considering a sample from a population, we usually take the central part of the distribution (or the central part of the bell curve). Because of this, the standard deviation calculated for the sample will be different (smaller) than the actual standard deviation for the entire population.

**Bessel's Correction** tries to fix the standard deviation calculation for the sample to match that of the population:
- `Variance = Σ( (x(i) - x̅)^2 ) / (n - 1)` where `n` is the size of the sample.
- `Standard deviation (s) = √ ( Σ( (x(i) - x̅)^2 ) / (n - 1) )`

**Note:** Bessel's Standard Deviation is represented by *Lowercase `s`*.

(Explanation: Instead of dividing by `n` for variance, we divide by `n-1`. Calculating the square root of  this new variance gives us the Standard Deviation - According to Bessel's correction).

Since the denominator is reducing by `1`, we get **bigger** numbers for Variance & Std. deviation `s`.

**Important:** When to use `s` and `σ`:
- If you have the population data and need to find the **standard deviation of the population** then we use `σ`.
- If you have a sample data set and need to **estimate the standard deviation of the population using the sample** then we use `s`.

**Note:** that for a small data set (which is **not** a sample estimating a population), we still use `σ`.

<!-- TOC --><a name="lessons-13-15-standardizing"></a>
# Lessons 13-15: Standardizing

Mean, Median and Mode give valuable data but there is some information that even these and standard deviation cannot help us with. For example, say you are a rated chess player and you want to know how many people you are better than as per your rating?

Depending on the distribution on the graph, we can tell what proportion or percentage of people you are better than at chess.

Percentage and proportions fall under the bracket of **Relative frequency**.

Up until now, we have been using the **Absolute frequencies** on the `Y-axis`. We can also picture **Relative frequencies** on the `Y-axis` by marking *proportion* or *percentage* frequency instead of the absolute/actual frequency values.

**The relative frequency distribution will look exactly similar to the absolute frequency distribution.**

<!-- TOC --><a name="finding-certain-proportion-of-the-sample-that-falls-between-two-bin-values"></a>
#### Finding certain proportion of the sample that falls between two bin values:
Using the relative frequency graph, we can **sum the relative frequencies that appear within the bins specified**. This sum gives us the **proportion** (or **percentage**) of the total sample that falls within the specified values. (Imagine doing this on a histogram).

<!-- TOC --><a name="disadvantage-of-relative-frequency"></a>
#### Disadvantage of Relative Frequency:
If we wanted to calculate the proportion of items falling within a range where the range limits are not the bin intervals (but fall within the bins) then it is difficult to be certain (Cannot give an exact proportion).

**Solution: Make the bin sizes smaller.** 

There is a drawback: Further decreasing the bin sizes can remove the shape of the distribution. (Think of bin sizes being so low that their frequencies are either 1 or 0 - rectangular-ish graph, shape of distribution lost)

<!-- TOC --><a name="continuous-distribution"></a>
### Continuous Distribution:
Instead of reducing the bin sizes, we can use a theoretical model. We map the distribution to a **theoretically continuous distribution** that represents the same (Shape is maintained). 

This distribution can be represented by mathematical formula. And, since it is continuous, it becomes easy to find the *proportions & percentages in more detail*.

(Visualize a continuous distribution: http://www.milefoot.com/math/stat/images/disccont.gif)

**Note:**
- In a continuous distribution, the *total area* under the curve must be equal to `1`.
- This follows from the fact that in a normal histogram too the relative frequencies of all bins add up to `1`.
- We can get *pretty good approximations* from the theoretical continuous distribution.

<!-- TOC --><a name="standardized-score"></a>
### Standardized Score:
Consider a **normal distribution** (`mean = mode = median = the peak`) . A continuous normal distribution would look something like this:
(https://onlinecourses.science.psu.edu/stat200/node/38)

Now consider this problem: How do we find how far or near a point (say, x) is from the mean(x̅) in terms of standard deviation(σ)? 

**Answer**: 
- x is `(x - x̅)/σ` standard deviations *above* x̅ if `x > x̅`
- x is `(x̅ - x)/σ` standard deviations *below* x̅ if `x < x̅`

**How does this value matter?**
Take an example of two people, one with number of facebook friends who are 3.5 standard deviations(σ) below the mean(x̅) number of facebook friends and the other with number of twitter followers who are 2.7 standard deviations(σ) below the mean(x̅) number of twitter followers. Who is more unpopular?

**Answer:** User one (with less facebook friends) is more unpopular because he deviates 3.5 times more than 
the standard deviation whereas user 2 deviates only 2.7 times it.

<!-- TOC --><a name="z-score"></a>
#### Z-score:
The Z-score ('zee score') is the number of standard deviations that any value is away from the mean.

`z = (x - x̅)/σ` where `x` is the value under consideration. (We always minus the mean from the value for z-score)

**Note:**
- If `z` is *negative*, the value is *smaller* than the mean.
- If `z` is *zero*, the value is equal to the mean.
- If `z` is *positive*, the value is *larger* than the mean.

**Negative z-score does *not* mean that the value itself is negative, and a Positive z-score does *not* mean that the value itself is positive (z-score cannot determine either of this)**.

<!-- TOC --><a name="meanx-in-terms-of-z-score"></a>
##### Mean(x̅) in terms of Z-score:
The mean(x̅) minus itself will return `0`. Therefore, z-score for mean(x̅) is `0`.

<!-- TOC --><a name="standard-deviation-in-terms-of-z-score"></a>
#### Standard Deviation(σ) in terms of Z-score:
One standard deviation will have a Z-score of `1`. How?

`X-axis value of 1σ = x̅ + σ` (assuming positive deviation)
`z = (x̅ + σ - x̅) / σ = (σ / σ) = 1`

Two Standard deviations: `z = 2` (positive) & `z = -2` (negative) (.. so on)

**Remember:**
- 65% of the data falls between 1 standard deviation of the mean.
- 95% of the data falls between 2 standard deviations of the mean.
- 99% of the data falls between 3 standard deviations of the mean.

If m = 1σ (One std. deviation) then how much is Nm (N times the std. deviation)? Answer: `(N * 1σ)`

<!-- TOC --><a name="standard-normal-distribution"></a>
### Standard Normal Distribution:
**Standard Normal Distribution** is a *Normal Distribution* with:
- a mean (central value) of `0` and 
- a standard deviation of `1`.

We can take any normal distribution, convert it to the standard normal distribution and scale it any way we want. For example, if we have facebook statistics from which we want to calculate the popularity of a person, we can do the following:

`Facebook friends = x = 210, mean(μ) = 190, σ = 36` **(Given)**

Therefore, `z-score of x = (210 - 190) / 36 = 0.56`

Using `z = 0.56`, we can come up with a new distribution for popularity by assuming a mean and a standard deviation. In this way, we can score the person based on popularity score.

`Popularity mean (pm) = 50, Popularity σ (pσ) = 10` **(say)**

Therefore, `Popularity score = pm + z*pσ = 50 + 0.56*10 = 55.6`

**Note:** 
- **Standard deviation** is *large* if the **spread** of the distribution is *large*.
- **Z-scores** of two values from two different normal distribution graphs (std normal distributions) can be *compared* (we can identify which one if further away from the mean than the other based on z-score)
- **Standard deviation** *cannot be negative* (We consider the absolute value). But, **Z-score** *can be negative*.

<!-- TOC --><a name="lessons-16-17-normal-distribution"></a>
# Lessons 16-17: Normal Distribution

Finding the area under the curve of a normal distribution - When using a continuous distribution (that theoretical curve). 

The area under the curve is called the **Probability Density Function(PDF)** written in equations as `f(x)` (We are not learning the equation).

In a Normal distribution, the tails actually *never* touch the x-axis but only get closer and closer to it. The tails then become **Horizontal Asymptotes**.

**Note:** The area under the curve represents the probability of a random value falling under that area (within the range of the values). 

Example: If there is a normal distribution of the heights of 7 year olds with mean 125 cm (x-axis = height, y-axis = relative frequency) then say the area below the value 120 is 0.14, the probability of picking a random girl whose height is less than 120 is also 0.14.

**(Important!)** Here, *area* and *proportion* values are similar. 
- Proportion of values less than a point x means: (Area from -infinity upto that point x).
- Proportion between two points x1 and x2 means: (Area from -infinity upto that point x2) - (Area from -infinity upto that point x1).

**(Important!)** Here, *area* and *percentile* values are similar. 
- If an (area covered by -infinity up to a point x) is `M` then the *percentile of x* is also `M`.

**Note:**
The graph we are looking at: 
- `x-axis` has z-scores or std. deviations(σ), and
- `y-axis` has the relative frequencies (proportion or percent).
The distribution is a *normal distribution* with z-scores (0 = mean) like in here: http://davidmlane.com/hyperstat/z_table.html

The total proportion (sum of proportions) of data values under the curve = `1` (Always = no matter the spread).

**Refresher**:
- `0.65` proportion of the data falls between 1 standard deviation of the mean.
- `0.95` proportion of the data falls between 2 standard deviations of the mean.
- `0.99`  proportion of the data falls between 3 standard deviations of the mean.

**Also**:
- `0.5` proportion of the data is *below* the mean(x̅).
- `0.5` proportion of the data is *above* the mean(x̅).

**Problem 1**: What is the proportion of values that fall either *below or above* the first std. deviation from the mean? 
**Answer**: `1 - 0.68 = 0.32`

**Problem 2**: What is the proportion of values that fall *below* the first std. deviation from the mean? 
**Answer**: `(1 - 0.68) / 2 = 0.16` (Normal distribution is symmetrical)

**Problem 3**: What is the proportion of values that fall *above* the second std. deviation from the mean? 
**Answer**: `(1 - 0.95) / 2 = 0.025` (Normal distribution is symmetrical)

**Problem 3**: What is the proportion of values that fall *above* the negative 2nd std. deviation but *below* the positive 1st std. deviation from the mean? 
**Answer**: `( 0.68 + ((0.95 - 0.68) / 2) ) = 0.815` (Normal distribution is symmetrical)

<!-- TOC --><a name="z-table"></a>
#### Z-table:
It is a table used to find the proportion of values between any points (including from infinity to a particular point) that is represented by a **Standard, Normal Curve** *Bell curve*.

Link to the Z-table: https://s3.amazonaws.com/udacity-hosted-downloads/ZTable.jpg

(On this table, `z` represents the z-score of the point that we are interested in (for mean = 0, 1std dev. = 1, 2std dev. = 2, etc) and gives us the *proportion of values from -infinity to that point*)

We know the proportions for values between 1st Std. deviation on either side(0.68), values between 2nd Std. deviation on either side(0.95), and so on. But, to calculate the proportions for any value we need to check the z-table (A formula also exists but we will not be learning it).

**The z-table gives the proportion of values *less* than a value with z-score `z`**

(**How to read the z-table: **https://classroom.udacity.com/courses/ud827/lessons/1478678538/concepts/1020887970923)

Example: Find the proportion of data *less* than 10 (mean = 6, std dev = 3)
Answer:
1 + (10-9)/3 = 1 + 1/3 = 1.33 = z-score. Use this `z` to look up the z-table.

Example: Find the proportion of data *greater* than 10 (mean = 6, std dev = 3)
Answer:
1 + (10-9)/3 = 1 + 1/3 = 1.33 = z-score. Use this `z` to look up the z-table. Now, after getting the proportion from the table, we need to subtract it from the total proportion (1) to get the answer (since z-table gives value less than a z-score but here we are finding for greater than).

Example: Find the proportion of data *between* 4 (mean = 6, std dev = 3) and 10.
Answer:
0 - (6-4)/3 = -2/3 = -0.66 = z-score = `z1`. Use this `z1` to look up the z-table. (say, `a1`).
1 + (10-9)/3 = 1 + 1/3 = 1.33 = z-score = `z2`. Use this `z2` to look up the z-table. (say, `a2`).
Final answer(proportion between values 4 & 10) = `a2 - a1`.

**An app that helps visualize the area under the curve:** http://psych.colorado.edu/~mcclella/java/normal/handleNormal.html


**Note**: Relationship between z-score & standard deviation is:

`z = (x - x̅) / σ` where `x` is the value of the point we are calculating z-score (z) for.

**Note**: For calculations for such problems (as above):
- *probability* is same as *area* which is same as *proportion(rel. freq.)*
- If an (*area* between -infinity up to a point x) is M then the *percentile* of that point x is also M

<!-- TOC --><a name="lessons-18-19-sampling-distributions"></a>
# Lessons 18-19: Sampling Distributions

We can compare the values in a sample with other values within the sample, we can compare values in a population with other values within the population. (We just solved for area, probability, proportion, percentile and percentage on a normal distribution curve).

**How do we compare one whole sample from another?**

**Expected Value**: a predicted value of a variable, calculated as the sum of all possible values each multiplied by the probability of its occurrence.

Example: In a tetrahedral die with values 1, 2, 3, and 4, the expected value or mean for the roll of the die is: 

`sum(eachpossibleoutcome*likelihoodofoccurring)` 

(**OR, also:** `sum of all the outcomes / number of outcomes`)

Each value can appear just as much as the occur (equal likelihood). 

`1 x 1/4 + 2 x 1/4 + 3 x 1/4 + 4 x 1/4 = 2.5`

`2.5` cannot possibly occur on a tetrahedral die. Therefore, the expected value *need not* be a value that should occur.

<!-- TOC --><a name="mean-of-the-sample-meansm"></a>
### Mean of the sample means(M):
The **mean of the sample means(`M`)** can be used to compare the means of the individual samples to know how far off or close it is to the population.

`M` is calculated by finding the average of all the sample means.

Distribution of sample = **sampling distribution** and it is the distribution of the means of all the samples in the population.

Mean of the sample means(Mean of the sampling distribution) is in the **center** and the **distribution is a Normal Distribution**.

It is tedious to calculate all sample means and find the mean of means (M) and sometimes even impossible to do so when the samples of the population are not discrete.

- Mean of *all values* in a population = `μ`
- Mean of *all the samples means* in a population = `M`
- And, importantly: `M = μ`

**Note:**
- Standard deviation of a `value from the mean` in a population/sample = `σ` 
- Standard deviation of a `sample mean` from the `mean of all sample means(M)` = `SE`  (*Std. Deviation of Distribution of Sample Means*)

<!-- TOC --><a name="relationship-between-se"></a>
#### Relationship between `σ` & `SE`:
`σ / SE = √n` where `n` is the sample size.

<!-- TOC --><a name="central-limit-theorem"></a>
### Central Limit Theorem:
The central limit theorem states that the sampling distribution of the mean of any independent, random variable will be normal or nearly normal, if the sample size is large enough.

By this theorem, the ratio of `σ` to `SE` will always be `√n`.

The `Std. Deviation` of sample means from `M` is represented by `se` and it is also called **Standard Error(SE)**.

**Remember:**
- Mean of the sample means(Mean of the sampling distribution) is in the **center** and the **distribution is a Normal Distribution**.

**Formulae**:
-  The **mean of the sampling distribution(M)** will be the same as the **mean of the population(μ)**
- `σ / SE = √n` where `n` is the sample size.

**Will increasing the sample size `n` decrease the spread of the normal distribution or not?**

Answer: It will. The reason is that by increasing `n`, `√n` denominator value will increase in `σ / √n = SE` and so the standard error decreases (the deviation from the mean of sample means decreases). Hence, we get *skinnier* graphs.

In conclusion: Greater the sample size `n` skinnier the graph (*inversely proportional*). In other words, **as the sample size (n) increases the standard error (SE) decreases**.

**Therefore: As the sample size (n) increases the sampling distribution gets skinnier**.

**Important: We can have any population of any shape and if we plot the distribution of its sample means, we get a normal distribution! (More and more normal as n increases) : Central Limit Theorem**

<!-- TOC --><a name="central-limit-theorem-revisited"></a>
#### Central Limit Theorem Revisited:
What the theorem says is:
- The distribution of sample means is approximately normal.
- The standard deviation (SE) of sample means is approx. `σ / √n`
- Mean of the sample means is equal to the population mean (μ).
