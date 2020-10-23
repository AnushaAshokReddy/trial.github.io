# Welcome To LearningApi Tutorial 

# Index 

<a href="#LearningApi_Introduction">LearningApi Introduction</a>

<a href="#LearningApi_Concept">LearningApi Concept</a>

<a href="#Algorithms&Modules">Introduction to Algorithms & Modules</a>

<a href="#Supported_Modules&Algorithms _List">Supported Modules and Algorithms</a>

<a href="#Example_Custom_Module&Algorithm">How to build the custom module and algorithm</a>

<a href="#Your_Contribution">Contribution to Learning API?</a>


# LearningApi Introduction <a id="LearningApi_Introduction"></a>

Learning API is Machine Learning Foundation of a set of ML libraries fully implemented as .NET Standard library. It provides a unique processing API for Machine Learning solutions. Because it is implemented fully in .NET, developers do not have to bridge .NET and Python or other popular ML frameworks. It has been developed in cooperation with daenet GmBh and Frankfurt University of Applied Sciences.

![Fig. 1]()

LearningAPI already has interfaces pre declared which we can easily access, understand and use in our project.

For example IAlgorithm, IScore, IResult, IPipeline module.

![Fig. 2]()

A small demo of the code is as followed : 

![Fig. 3]()

LearningAPI is a foundation of Machine Learning algorithms, which can run in the pipeline of modules compatible to each other. This concept allows using of conceptually different algorithms in the same API, which consists of a chain of modules. One pipeline module is defined as implementation of interface IPipeline. 

The format for IPipeline : 
```markdown
 public interface IPipelineModule
    {

    }

 public interface IPipelineModule<TIN, TOUT> : IPipelineModule
    {
        TOUT Run(TIN data, IContext ctx);
    }
```

# The LearningApi Concept <a id="LearningApi_Concept"></a>

LearningAPI already has interfaces pre declared which we can easily access, understand and use in our project.

  In order use LearningApi, we should install **_Nuget packages_** and **_LearningApi_** into our project. 
  
  Basically a NuGet package is a single ZIP file with the *.nupkg* extension that contains compiled code (DLLs), other files related to that code, and a descriptive manifest that includes information like the package's version number.
  
  Initially open the class ‘.cs’ and implement the IAlgorithm in the code which is taken from Learning Api NuGet package. IAlgorithm is in the library and it has a separate structure which we have to use in the project. 

  A NuGet package is a single ZIP file with the .nupkg extension that contains compiled code (DLLs), other files related to that code, and a descriptive manifest that includes information like the package's version number. The NuGet package 'LearningApi' enables the end users to use all the predefined interfaces facilitating the defined functionalities and make it easier for the end user to achieve the goal of creating a solution.
  
More information can be found on [Click here for more information on NuGet packages..](https://docs.microsoft.com/en-us/visualstudio/mac/nuget-walkthrough?view=vsmac-2019)

To find out more details, click on [Information..](https://docs.microsoft.com/en-us/nuget/what-is-nuget)

**IAlgorithm** - The structure of _IAlgorithm_ has 2 phases:

1. _**IResult**_ – IResult is used to set and get the final result of the algorithm and store it. We use IResult for the PREDICT phase - This is the final phase where we get the perfect output for the input provided by the user on the basis of the algorithm we give.In this prediction logic should be written as shown in  screenshot 6.

2. _**IScore**_ – Iscore is used to set and get the values of the variables used in the project. We use IScore for RUN and TRAIN methods.

**RUN** – This is the evaluation part where the random data will be given to our system to test whether the correct output is being displayed after the training session. Here, we call TRAIN method internally.

**TRAIN** – Here we will train the system with our specific set of data I.e input and the output as in how to function. Algorithm of the function is written in Train method.

  **Inputs** to the TRAIN i.e to the algorithm is the set of data with expected outputs for few number of inputs, we train the system and then expect the predicted value to be accurate when other input is given.
  
  **Output** is the predicted value from PREDICT method which gives the accuracy of the True or False statements.
  
**The Pipeline module** receives an input TIN and context information. Usually TIN is set of data, which results as output of th eprevious module. Typically, first module in the pipeline is responsibe to provide learning data and last module in the pipeline is usually algorithm.

Following example illustrates how to setup the learning pipeline:

```markdown
public void SimpleSequenceTest()
        {
            LearningApi api = new LearningApi();
            api.UseActionModule<object, double[][]>((notUsed, ctx) =>
            {
                const int maxSamples = 10000;
                ctx.DataDescriptor = getDescriptor();
                double[][] data = new double[maxSamples][];

                //
                // We generate following input vectors: 
                // IN Val - Expected OUT Val 
                // 1 - 0
                // 2 - 0,
                // ...
                // maxSamples / 2     - 1,
                // maxSamples / 2 + 1 - 1,
                // maxSamples / 2 + 2 - 1,


                for (int i = 0; i < maxSamples; i++)
                {
                    data[i] = new double[2];
                    data[i][0] = i;
                    data[i][1] = (i > (maxSamples / 2)) ? 1 : 0;
                }

                return data;
            });

            api.UsePerceptron(0.02, 10000);

            IScore score = api.Run() as IScore;

            double[][] testData = new double[4][];
            testData[0] = new double[] { 2.0, 0.0 };
            testData[1] = new double[] { 2000.0, 0.0 };
            testData[2] = new double[] { 6000.0, 0.0 };
            testData[3] = new double[] { 5001, 0.0 };

            var result = api.Algorithm.Predict(testData, api.Context) as PerceptronResult;

            Assert.True(result.PredictedValues[0] == 0);
            Assert.True(result.PredictedValues[1] == 0);
            Assert.True(result.PredictedValues[2] == 1);
            Assert.True(result.PredictedValues[3] == 1);
        }
        
         private DataDescriptor getDescriptor()
        {
            DataDescriptor desc = new DataDescriptor();
            desc.Features = new LearningFoundation.DataMappers.Column[1];
            desc.Features[0] = new LearningFoundation.DataMappers.Column()
            {
                Id = 0,
                Name = "X",
                Type = LearningFoundation.DataMappers.ColumnType.NUMERIC,
                Index = 0,
            };

            desc.LabelIndex = 1;

            return desc;
        }
```
The code shown above setups the pipeline of two modules. 

1.First one is so called action module, which defines the custom code to be executed. 

2.Second module is setup by the following line of code:

```markdown
api.UsePerceptron(0.02, 10000);
```
It injects the perceptron algorithm in the pipeline.

Execution of the pipeline is started with following line of code:

```markdown
IScore score = api.Run() as IScore;
```

When the pipeline starts, modules are executed in the sequenceordered as they are added to the pipeline. 
In this case, first action module will be executed and then perceptron algorithm. After running of the pipeline model is trained. Next common step in Machine Learning applications is called evaluation of the model. Following code in previous example shows how to evaluation (predict) the model:

```markdown
double[][] testData = new double[4][];
            testData[0] = new double[] { 2.0, 0.0 };
            testData[1] = new double[] { 2000.0, 0.0 };
            testData[2] = new double[] { 6000.0, 0.0 };
            testData[3] = new double[] { 5001, 0.0 };

        var result = api.Algorithm.Predict(testData, api.Context) as PerceptronResult;

            Assert.True(result.PredictedValues[0] == 0);
            Assert.True(result.PredictedValues[1] == 0);
            Assert.True(result.PredictedValues[2] == 1);
            Assert.True(result.PredictedValues[3] == 1);
```

# Introduction to Algorithms & Modules <a id="Algorithms&Modules"></a>

An algorithm is a set of logical coding which is trained with lots and lots of data to predict the otput most accurately.

A module in Machine Learning represents a set of code that can run independently and perform a machine learning task, given the required inputs. A module might contain a particular algorithm, or perform a task that is important in machine learning, such as missing value replacement, or statistical analysis.
Both algorithms and modules are independent of each other. 

While implementing an algorithm, it is initially trained using various number of data available already to make the algorithm learn hpw to predict the results for an unknown input in the later stages. Thus the set of data is very important. 

These data is supposed to be very clean with all details. Sometimes in algorithms when we dont get clean data, pipeline modules are used for pre-processing of the data. For example some pipeline modules as MinMaxNormalisers have the function of normalising the data for the bigger algorithm. 


# Supported Modules and Algorithms <a id="Supported_Modules&Algorithms _List"></a>


All the supported Modules and Algorithms are listed in an excel sheet. Also, the information about the documentation and coding source files availabiliy in the LearningApi repository can be found here.

[Click here to find the list..](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/blob/master/LearningApi/src/AlgorithmsModules%20_list.xlsx)


# How to build the custom module and algorithm <a id="Example_Custom_Module&Algorithm"></a>

  The below solution demonstrates how to implement a custom module and a custom algorithm. In this example, the SUM and AVERAGE logic will be impemented.
  
  The below example is only for reference on steps to implement a solution using LearningApi and the pipeline modules. But there is no need to build both these for every project.
  
  Pipeline components are not ML algorithms. They are processing components in the middle of pipeline. They can be used to load data, to filter data or to do some pre-calculations.So, only sometimes, we use Pipieline modules in MLAlgorithms for processing of clean data and not everytime we should implement this in algorithm.
	
## Algorithm

Prediction of 'chance of Precipitation' by calculating the average of temperature data and the average of chance of precipitation till date given. Motive is to achieve the solution using LeaningApi framework.

## Example Solution using LearningApi Algorithm :

### Step 1: Create a solution 

In the Visual Studio, create a new solution by following the steps -
	
  ```markdown Navigate to File --> New --> Project
  ```
For our example - give the project name as **“HelloWorldTutorial”**

Use the selectors on the left side to choose the different types of programming languages or platforms to work with. For example, we are creating a class library with the template .NET STANDARD under the Visual C# selector as show in Fig. 4.

```markdown Name the project --> Specify the location --> Click OK<
```
Now the project is created with the name _'HelloWorldTutorial.sln'_
  
![Fig. 4 : Creation of Solution]()
	
### Step 2: Create the class library for the algorithm 

When solution(HelloWorldTutorial.sln) is created, by default a class library is also created automatically.

We have to change the names accordingly. Here for example, I’m leaving my project name as “HelloWorldTutorial” but changing the class library name as “LearningApiAlgorithm.cs” as shown in Fig. 5.

LearningApiAlgorithm.cs serves as the main class folder for the algorithm.

![Fig. 5 : The project and class library folder structure]()
	
### Step 3 : Add NuGet Package 'LearningApi' to our project 

We should add NuGet package called LearningApi to our project by following the steps below, 

		
	Right click on project (HelloWorldTutorial.sln) --> Click on ‘Manage NuGet packages..’

	in the pop up window --> Click on BROWSE,
	
	search for LearningApi and select --> Click on SELECT button to add the NuGet package to the project. (Refer Fig. 6)
	

![Fig. 6 : Adding NuGet package to our project]()
  
A pop up with the packages installed along with the LearningApi NuGet package is displayed. Click on OK button.

### Step 4 : Start the Code for the project LEARNING API ALGORITHM

Open the class *‘LearningApiAlgorithm.cs’* and implement the *IAlgorithm* in the code which is taken from LearningApi NuGet package. *IAlgorithm*  is in the library and it has a separate structure which we have to use in the project as we already have discussed in the section - 'how to use LearningApi'. 

![Image 7]()

Fig. 7 : Structure of IAlgorithm integrated

![Image 8]()

Fig. 8 : Algorithm of the project is included in *TRAIN* method of IScore

![Image 9]()

Fig. 9 : Prediction logic is included in *PREDICT* method of IResult

### Step 5 : Create the *Extension.cs* file 

Extension file in a project facilitates other users to utilise our project code in their implementations. Calling this file in other projects enables the project code in other projects.

In our example solution, *LearningApiAlgorithmExtension.cs file* is created in our example solution to demonstarte the structure of the file as shown in Fig. 10.

![Image 10]()

Fig. 10 : Extension file structure and usage shown

### Step 6 : Create the *Result.cs* and *Score.cs* files 

In our example solution, the *LearningApiAlgorithmResult.cs* and *LearningApiAlgorithmScore.cs* files should be created to define the values which should be storing the result and trained score data.

![Image 11]()

Fig. 11 : Structure of solution with Learning API Algorithm.

-------------------------------------------------------------------------------------------------------------
## Example Solution using LearningApi Pipeline Module :

Pipeline module is a canal to send the data to the actual Algorithm. For a deeper knowledge on Pipeline Module click on <a href="#Algorithms&Modules">Click Here..</a>

### Step 7 : Create LearningAPI Module for the Pipeline module method

This does not have any structure and we won’t pass any algorithm or logic. And also an Extension.cs file is created for this to facilitate the other users to use this method.

![Image 12]()

Fig. 12 : Structure of the Pipeline module 

## Result of Algorithm 

According to the algorithm, the set of data of temperature is given and taken the average of the temperature. The data for chance of precipitation is taken an average. The ratio of average of temperature and average of chance of precipitation is given to be our score. When this score is multiplied with each data given, we get the precipitation value predicted.

# How can you contribute to Learning API? <a id="Your_Contribution"></a>

If you have implemented a custom module or algorithm and want to integrate it to LearningAPI, then you can do the following, 

- Contact us - implement a page for this 
- Implement your algorithm or/and module
- create the pull request
- Create an issue in the Repository


