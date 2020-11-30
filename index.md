[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/blob/master/LICENSE)

<button>Default Button</button>
<a href="#https://anushaashokreddy.github.io/trial.github.io/Basic_Example_Sum.md" class="button">Anusha Button</a>
<button class="button">Button</button>
<input type="button" class="button" value="Input Button">


[Part 2 - Linear_Regression](https://anushaashokreddy.github.io/trial.github.io/Linear_Regression_Example)

<a href="https://anushaashokreddy.github.io/trial.github.io/Linear_Regression_Example"><span class="button">Click this button</span></a>

# Welcome To LearningApi Tutorial 

# Index 

<a href="#LearningApi_Introduction">LearningApi Introduction</a>

<a href="#Supported_Algorithms&Modules_List">Supported Algorithms and Modules</a>

<a href="#LearningApi_Concept">The LearningApi concept</a>

<a href="#What_is_Module">What is a Learning API Pipeline Module?</a>

<a href="#What_is_Algorithm">What is a Learning API algorithm?</a>

<a href="#Example_Custom_Algorithm">How to build the custom algorithm?</a>

<a href="#Your_Contribution">Contribution to Learning API?</a>


# LearningApi Introduction <a id="LearningApi_Introduction"></a>

_LearningApi_ is a Machine Learning Foundation of a set of ML algorithms implemented in .NET Core/C#. It provides a unique pipeline processing API for Machine Learning solutions. Because it is implemented fully in .NET, developers do not have to bridge .NET and Python or other popular ML frameworks. It has been developed in cooperation with daenet GmbH and Frankfurt University of Applied Sciences.

<!--
![Image 1](https://user-images.githubusercontent.com/44580961/98464210-a5dc1200-21c1-11eb-95ef-e1a0d7942382.png)--> 

<!-- Fig. 1 : daenet GmbH and Frankfurt University of Applied Sciences --> 

LearningAPI already has interfaces that are pre declared which we can easily access, understand and use in our project.

Before you start with the *LearningAPI*, you should get familiar with several interfaces: IPipeline Module, IAlgorithm, IScore, IResult. These interfaces are shared across all algorithms inside of the *LearningAPI*.

LearningAPI is a foundation of Machine Learning algorithms, which can run in the pipeline of modules compatible to each other. This concept allows using of conceptually different algorithms in the same API, which consists of a chain of modules. Typically in Machine Learning applications, developers need to combine multiple algorithms or tasks to achieve a final task or result.

For example, imagine you want to train a supervised algorithm from historical power consumption data to be able to predict the power consumtion. The training data is stored in a csv file which can be read into the program for training the model. This csv file needs to be described in DataDescriptor interface. It contains features like outside temperature, the wind and power consumtion (where power consumption is the output prediction value which is called as label). To solve the problem, you first have to read the data from CSV, then to normalize features and then to train the algorithm.

You could think about these tasks as follows:

1. Read CSV
2. Normalize the data
3. Train the data.

After the above process, you will have a trained instance of the algorithm *algInst*, which can be used for prediction:

4. Use *algInst* to predict the power consumption based on the given temperature and wind.

To solve this problem with the *LearningAPI* the following pseudo can be used:

```
var api = new LearningApi(config)
api.UseCsvDataProvider('csvFileName.csv')
api.UseNormilizerModule();
api.Train();
// Prediction
var predictedPower = api.Predict(108W, 45 wind force);
```

We can implement the solution for the above discussed model using _pipeline_ method. A pipeline is defined as the list of pipeline modules. One pipeline module is defined as implementation of interface _IPipeline_.

The IPipeline Interface is defined as follows:

```csharp
 public interface IPipelineModule
 {
 }
 public interface IPipelineModule<TIN, TOUT> : IPipelineModule
 {
        TOUT Run(TIN data, IContext ctx);
 }
```

With this definition the developer can run the implementation of the module with the following code:

```csharp
using LearningFoundation;
public interface IAlgorithm : IPipelineModule<double[][], IScore>, IPipelineModule
{
	IScore Train(double[][] data, IContext ctx)
  	IResult Predict(double[][] data, IContext ctx)
}
```

To define the pipline of modules you woudld typically do describe the following modules differnetly in the same file and call them wherever needed to do the respective functions:

api.UseDataProviderModule(“DataProvider”, DataProviderModule)
api.UseDataNormalizerModule(“DataNormalizer”, DataNormalizerModule);
 
To make the code more readable, developers of modules typically provide helper extension methods using the following ,	
 
api.UseDataProvider(args1)
api.UseDataNormalizer(args2);

-------------------------------

A real time example model is explained in the below section -
<a href="#Example_Custom_Algorithm">'Please click here to understand 'How to build a LearningAPI algorithm?'</a>

-------------------------------


# Supported Algorithms and Modules <a id="Supported_Algorithms&Modules_List"></a>

All the supported Modules and Algorithms are listed in an excel sheet. Also, the information about the documentation and coding source files availabiliy in the LearningApi repository can be found here.

[Click here to find the list in Git Repository..](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/blob/master/LearningApi/src/AlgorithmsModules%20_list.xlsx)

## Machine Learning Algorithms

| Algorithm | LearningApi Repository | .md file available | Documentation available? |
|:--- |:--- |:--- |:--- |
| RingProjection | [Github_RingProjection Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/RingProjection) | Available | [Github_RingProjection Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/RingProjection/Documentation) |
| SVM - SupportVectorMachine | [Github_SVM Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/SupportVectorMachine)    | Available | [Github_SVM Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/SupportVectorMachine/Documentation) |
| Scalar encoder - ScalarEncoder in HTM  | [Github_ScalarEncoder Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/ScalarEncoder%20in%20HTM) | Not Available yet | [Github_ScalarEncoder Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/ScalarEncoder%20in%20HTM/Documentation) |
| Anamoly latest - AnomDetectLatest | [Github_Anamoly Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/AnomDetectLatest) | Available | [Github_Anamoly Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/AnomDetectLatest/Documentation) |
| Delta Learning | [Github_DeltaLearning Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/DeltaRuleLearning) | Available | [Github_DeltaLearning Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/DeltaRuleLearning/Documentation) |
| GaussianMean Filter | [Github_GaussianMeanFilter Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/GaussianMeanFilter) | Available | [Github_GaussianMeanFilter Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/GaussianMeanFilter/Documentation) |
| Image Edge detection    | [Github_ImageEdgeDetection Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/ImageDetection) | Available  | [Github_ImageEdgeDetection Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/ImageDetection/Documentation) |
| Logistic Regression | [Github_Logistic Regression Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/LogisticRegression) | Available | [Github_LogisticRegression Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/LogisticRegression/Documentation) |
| Neural Network Perceptron | [Github_NeuralNetworkPerceptron Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/NeuralNetworks) | Not Available yet | [Github_NeuralNetworkPerceptron Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/NeuralNetworks/NeuralNet.MLPerceptron/Documentation)    |
| Self Organizing Map | [Github_SelfOrganizingMap Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/SelfOrganizingMap) | Available | [Github_SelfOrganizingMap Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/SelfOrganizingMap/Documentation) |
| Survival Analysis | [Github_SurvivalAnalysis Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/MLAlgorithms/SurvivalAnalysis) | Not Available yet | Not Available yet |

## Data Modules

| Modules | LearningApi Repository | .md file available | Documentation available? |
|:--- |:--- |:--- |:--- |
| Image binarizer Latest | [Github_ImageBinarizer Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizerLatest) |  Available | [Github_ImageBinarizer Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizerLatest/Documentation) |
| Euclidian color filter | [Github_EuclidianColorFilter Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter) | Available | [Github_EuclidianColorFilter Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter/Documentation) |
| Image Binarizer | [Github_ImageBinarizer Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer) | Available  | [Github_ImageBinarizer Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer/Documentation) |
| Center Module | [Github_CenterModule Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module) | Available | [Github_CenterModule Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module/Documentation) |
| Canny edge detector | [Github_CannyEdgeDetector Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/CannyEdgeDetector) | Available    | Not Available yet |
| SDR Classifier | [Github_SDR Classifier Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier) | Available | [Github_SDR Classifier Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier/Documentation) |

# The LearningApi Concept <a id="LearningApi_Concept"></a>

LearningAPI is a framework for developing software machine learning applications. This includes predefined classes and functions that can be used to process input, train the system and give an accurate predicted answer.

  In order to use LearningApi, we should install the Nuget package called **_LearningApi_** into our project (this will be demonstarted in <a href="#Example_Custom_Algorithm">Example custom algorithm section</a>)
  
  Basically a NuGet package is a single ZIP file with the *.nupkg* extension that contains compiled code (DLLs), other files related to that code, and a descriptive manifest that includes information like the package's version number.
  
  Initially open the class ‘.cs’ and implement the IAlgorithm in the code which is taken from LearningApi NuGet package. IAlgorithm is in the library and it has a separate structure which we have to use in the project. 
  
More information can be found on [Click here for more information on NuGet packages..](https://docs.microsoft.com/en-us/visualstudio/mac/nuget-walkthrough?view=vsmac-2019)

<!--To find out more details, click on [Information..](https://docs.microsoft.com/en-us/nuget/what-is-nuget)-->

  **Inputs** to the TRAIN i.e to the algorithm is the set of data with expected outputs for few number of inputs, we train the system and then expect the predicted value to be accurate when other input is given.
  
  **Output** is the predicted value from PREDICT method which gives the accuracy of the True or False statements.
  
  For example, if we take HOUSE PRICE prediction scenario ( <a href="#Model_Explanation">(Click here for model explanation)</a>), the features SIZE, ROOM and PRICE  are the real time _input data_ given to the model to get trained based on these existing data. Whereas , PRICE is the predicted value which is expected to be the output of the model based on the training given to the model. 

**IAlgorithm** - The _IAlgorithm_ interface has 2 phases:

1. _**IResult**_ – IResult is used to set and get the final result of the algorithm and store it. We use IResult interface for the PREDICT phase - This is the final phase where we get the accurate predicted output for the input provided by the user on the basis of the trained model. IResult is returned by PREDICT. 

2. _**IScore**_ – Iscore is used to set and get the values of all the features used in the project (which ar given in csv file/input data). IScore is returned by RUN / TRAIN methods.

**RUN/TRAIN** – This is the training (learning) part where the random/real time data will be given to our system to test whether the correct output is being displayed after the training phase. The description of features, data inputs, logic for learning are defined in this interface. Here we will train the system with our specific set of data i.e input and the output as in how to function/ predict the output with higher accuracy.

**PREDICT** –  This is the prediction part where the model has the trained logic as input and gives the high accuracy prediction for the label we described to be as output. 
  
**The Pipeline module** receives an input TIN and context information. Usually TIN is set of data, which results as output of th eprevious module. Typically, first module in the pipeline is responsibe to provide learning data and last module in the pipeline is usually algorithm.

***Implement a button - Button name - Pipeline Module ***

<button>Pipeline Module</button>
<a href="#" class="button">Pipeline Module</a>
<button class="button">Pipeline Module</button>

# What is a LearningApi Pipeline Module? <a id="What_is_Module"></a>

A module in Machine Learning represents a set of code that can run independently and perform a machine learning task, given the required inputs. A module might contain a particular algorithm, or perform a task that is important in machine learning, such as missing value replacement, or statistical analysis.
Both algorithms and modules are independent of each other. 

While implementing an algorithm, it is initially trained using various number of data available already to make the algorithm learn how to predict the results for an unknown input in the later stages. Thus the set of data is very important. This data is supposed to be clean with all details. Sometimes in algorithms when we don't get clean data, pipeline modules are used for pre-processing of the data. 

For example some pipeline modules as MinMaxNormalisers have the function of normalising the data for the larger algorithms.

Following example illustrates how to setup the learning pipeline modules:

```csharp
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
The code shown above sets up the pipeline for two modules. 

1.First one is so called _Action Module_, which defines the custom code (to generate the vector functions) to be executed in the program to achieve . 
```csharp
api.UseActionModule<object, double[][]>((notUsed, ctx)
```

2.Second module injects the perceptron algorithm in the pipeline and it is setup by the following line of code:

```csharp
api.UsePerceptron(0.02, 10000);
```

3.Execution of the pipeline is started with following line of code:

```csharp
IScore score = api.Run() as IScore;
```
4.DataDescriptor part is to define the input data we use.The below lines of code guides the program what input should be considered. 

```csharp
desc.Features[0] = new LearningFoundation.DataMappers.Column()
            {
                Id = 0,
                Name = "X",
                Type = LearningFoundation.DataMappers.ColumnType.NUMERIC,
                Index = 0,
            };

            desc.LabelIndex = 1;
```

When the pipeline starts, modules are executed in the sequence ordered as they are added to the pipeline. 
In this case, first 'Action Module' will be executed and then 'Perceptron' algorithm. After running of the pipeline modules, model is trained. Next common step in Machine Learning applications is called evaluation of the model. Following code in previous example shows how to evaluate (predict) the model:

```csharp
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

# How to build the custom module? <a id="Example_Custom_Module"></a>

The below solution demonstrates how to implement a custom pipeline module. In this example, convolution logic will be impemented.
  
This example is only for reference on steps to implement a solution using LearningApi. 

## Example Solution using LearningApi Pipeline Module :

Pipeline module is a canal to send the data to the actual Algorithm. For a deeper knowledge on Pipeline Module click on <a href="#WModule">Click Here..</a>

Let's implement Pipelinemodule for a Convolution Filter.

### Step 1: Create a solution for Pipeline module

This does not have any particular structure and we won’t pass any major algorithm here. 

In the Visual Studio, create a new solution by following the steps -
	
    Navigate to File --> New --> Project/New Solution

Use the selectors on the left side to choose the different types of programming languages or platforms to work with. For example, we are creating a class library with the template .NET STANDARD under the Visual C# selector as show in Fig. 4.

    Click on NEXT 	

<!--![Image 4](https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png" width="600" height="450" />

Fig. 16 : New Project

For our example - given the project name as **“HelloLearningApiPipelineModule”**	

    Name the project --> Solution Name --> Specify the location --> Click OK/CREATE
    
<!--![Image 17](https://user-images.githubusercontent.com/44580961/98464517-bf7e5900-21c3-11eb-9a71-1d03adfea118.png)--> 

<img src="https://user-images.githubusercontent.com/44580961/98464517-bf7e5900-21c3-11eb-9a71-1d03adfea118.png" width="600" height="450" />

Fig. 17 : Project and Solution name

Now the project is created with the name _'HelloLearningApiPipelineModule.sln'_
  
<!--![Image 18](https://user-images.githubusercontent.com/44580961/98464519-c2794980-21c3-11eb-81e7-dee1ccd54601.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464519-c2794980-21c3-11eb-81e7-dee1ccd54601.png" width="500" height="300" />

Fig. 18 : Creation of Solution	

### Step 2: Create the class library for the module 
	
When solution(HelloLearningApiPipelineModule.sln) is created, by default a class library is also created automatically (.cs file).

Change the class library name as “HelloLearningApiPipelineModule.cs” and also create a nwe class withe name 'HelloLearningApiPipelineModuleExtension' as shown in Fig. 19.

![Image 19](https://user-images.githubusercontent.com/44580961/98464522-c60cd080-21c3-11eb-85cf-dea9250c2e4d.png) 

Fig. 19 : Pipeline and Extension class files

### Step 3 : Add NuGet Package 'LearningApi' to our pipeline module project 

We should add NuGet package called _LearningApi_ to our project by following the steps below, 

		
	Right click on project (HelloWorldTutorial.sln) --> Click on ‘Manage NuGet packages..’ 

	in the pop up window --> Click on BROWSE, 
	
	search for LearningApi and select --> Select the checkbox of LearningApi nuget --> Click on SELECT/ADD PACKAGE button 

<!--![Image 20](https://user-images.githubusercontent.com/44580961/98464524-ca38ee00-21c3-11eb-9542-c05a6e9922f1.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464524-ca38ee00-21c3-11eb-9542-c05a6e9922f1.png" width="400" height="300" />

Fig. 20 : Nuget package added to pipeline project

### Step 5 : Implement IPipeline Module 

Ipipeline Module from LearningApi should be integrated in the Module coding as shown in the Fig. 21.

![Image 21](https://user-images.githubusercontent.com/44580961/98464526-cc02b180-21c3-11eb-9e2d-1afa8e1f86d2.png)

Fig. 21 : IPipeline module Interface in example module

### Step 6 : Coding for the example pipeline module logic of convolution filter 

This is not a major algorithm, instead a small pre processing of Convolution filter which can be used for any other algorithms as the data. Code format is as shown below, 

![Image 22](https://user-images.githubusercontent.com/44580961/98464527-d02ecf00-21c3-11eb-81e8-6180c1901fac.png)

Fig. 22 : IPipeline module for example module

## Result of Module 

Explain result and give picture here.

***Implement a button - Button name - Pipeline Example code in Github ***

<button>Pipeline Example code in Github</button>
<a href="#" class="button">Pipeline Example code in Github</a>
<button class="button">Pipeline Example code in Github</button>

-------------------------------------------------------------------------------------------------------------------------------------
***Implement a button - Button name - Algorithm ***

<button>Algorithm</button>
<a href="#" class="button">Algorithm</a>
<button class="button">Algorithm</button>

# What is Algorithm? <a id="What_is_Algorithm"></a>

Machine learning is a class of methods for automatically creating models from data. Machine learning algorithms are the engines of machine learning, meaning it is the algorithms that turn a data set into a model. Which kind of algorithm works best (supervised, unsupervised, classification, regression, etc.) depends on the kind of problem you’re solving, the computing resources available, and the nature of the data.

An algorithm is a set of logical coding which is trained with lots and lots of data to predict the otput most accurately.

# How to build the custom algorithm using LearningAPI? <a id="Example_Custom_Algorithm"></a>

  The below solution demonstrates how to implement a simple model SUM calculation using LearningApi. To understand the implementation, you should initially understand the logic we are using. 
	
To train a model to learn predicting the calculation of POWER consumption by doing SUM calculation of 2 variables like SOLAR and WIND, we need to train the model with few real example data and then train it to learn the logic in order to give the accurate output. 

Below is the explanation of implementation of LearningApi concept for building the SUM model as explained earlier. 

We have also given the similar guidance for an actual Machine Learning Algorithm at the end of this section. 

## LearningApi Example Algorithm <a id="#Example_Algoirthm"></a>

Let's take a simple example to build a model to predict the power consumption using the given set of data. For this, we have 3 phases of tasks to perform. 

1. Loading data phase
2. Training Phase
3. Prediction Phase 

**Loading data phase :** 

Let's consider 2 variables which are refered as _features_ in ML (features are the inputs)
	
	*SOLAR* - Solar data in kWh unit
	
	*WIND* - Wind data in knot unit
	
and 1 _label_ (the output feature is called label)

	*POWER* - Power consumption data 

Each features is given with an index number defined in DataDescriptor as shown in the following line of code :

```csharp
	des.Features[0] = new Column { Id = 1, Name = "solar", Index = 0, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
        des.Features[1] = new Column { Id = 2, Name = "wind", Index = 1, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
        des.Features[2] = new Column { Id = 3, Name = "power", Index = 2, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
```
	
Now we should train the model with the above 2 features and a label by giving a real data (shown below) in csv file, we have taken the following data :

| solar | wind | power | 
|:--- |:--- |:--- |
| 10 | 5 | 15 |
| 20 | 8 | 28.4 |
| 25 | 2 | 27 |
| 34 | 10 | 44 |
| 40 | 20 | 60 |
| 42 | 2 | 44 |
| 45 | 40 | 85 |
| 48 | 2 | 50 |
| 50 | 20 | 70 |
| 55 | 25 | 80 |

The model uses CsvDataProvider to read the csv file as shown in the following interface :

```csharp
 api.UseCsvDataProvider(trainDataPath, ',', false);
```

The module can be configured by using a DataDescriptor class. This class describes the columns (features and the label). Label index is the index number of feature for which the prediction should be made. 

The following is the code snippet for the data descriptor :

```csharp
 private DataDescriptor LoadMetaData()
        {
            var des = new DataDescriptor();

            des.Features = new Column[3];
	    
	    //.....(DESCRIBE THE COLUMNS HERE)

            des.LabelIndex = 2;
            return des;
        }
```

**Training phase :** The model reads the data from _csv_ file and based on the data given, after training, the model should be able to predict the result y. 

By seeing the data set given, we can easily recognise that power consumption is the result of summing up solar and wind data. The simple logic we are using here is SUM : 

The expression for the problem can be identified as: 

	POWER = SOLAR + WIND 

For this basic example, learning is not required as the data is straight forward. Model predicts the power consumption value using the data and most of the time loss we get would be zero. 

**1st data : solar 10 and wind value 5 **

	substitute these values in formula --   y = solar + wind
					       y1 = 10 + 5 = 15	      
	
	Square Error -- SE1 = (actual power value for above solar and wind values - predicted price for above solar and wind values)^2 
	
			SE1 = (15 - 15)^2 = 0^2 = 0 
			
But in some cases, there might be outliers like shown below :
			
**2nd data : solar 20 and wind value 8 **

	substitute these values in formula --   y = solar + wind
					       y2 = 20 + 8 = 28
	
	Square Error -- SE2 = (actual power value for above solar and wind values - predicted price for above solar and wind values)^2
	
			SE2 = (28.4 - 28)^2 = 0.4^2 = 0.16
			
The above calculation occurs for all the values of solar and wind data set.

We use *Mean Square Error* concept to find out the error differences and help the model to finalise the least error value to be more accurate for the prediction. For each data set MSE is calculated.

Mean square error calculation (MSE1)-- 

    	     = (SE1+SE2+....)/ (Total number of data)
    
	MSE1 = x

The following code snippet shows the calculating logic in the program :
```csharp
for (int trainDataIndex = 0; trainDataIndex < numTrainData; trainDataIndex++)
            {
                estimatedOutputLabels[trainDataIndex] = ComputeSum(inputFeatures[trainDataIndex]);
                squareErrors[trainDataIndex] = ComputeSquareError(actualOutputLabels[trainDataIndex], estimatedOutputLabels[trainDataIndex]);
            }

            double meanSquareError = squareErrors.Sum() / numTrainData;

            loss = meanSquareError;

            if (ctx.Score as SumScore == null)
                ctx.Score = new SumScore();

            SumScore scr = ctx.Score as SumScore;
            scr.Loss = loss;

            return ctx.Score;
```

While the loss is very less, model gives the accurate power consumption value for the given set of solar and wind data for test. 
	
**Prediction Phase :** Now, the model has an exact equation (found out as the output of training with least loss).  This formula/logic will be used by the model to predict power consumption with any further solar and wind combinations.

## Implementation of LearningApi for the above example of SUM Algorithm :

This topic provides a deep knowledge on implementation of LearningApi for the algorithm we discussed above. Here main focus is to learn where and how we implement the LerningApi interfaces in the project and not the algorithm itself.

### Step 1: Create a solution 

In the Visual Studio, create a new solution by following the steps -
	
    Navigate to File --> New --> Project

Use the selectors on the left side to choose the different types of programming languages or platforms to work with. For example, we are creating a class library with the template .NET STANDARD under the Visual C# selector as show in Fig. 4.

    Click on NEXT 	

<!--![Image 4](https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png" width="600" height="450" />

Fig. 4 : New Project

For our example - given the project name as **“SumAlgorithm”**	

    Name the project --> Solution Name --> Specify the location --> Click OK/CREATE
    
<!--![Image 5]() -->

<img src="https://user-images.githubusercontent.com/44580961/100554990-cbd76c80-32be-11eb-8f0e-2b203e669ea0.png" width="600" height="450" />

Fig. 5 : Project and Solution name

Now the project is created with the name _'SumAlgorithm.sln'_
  
<!--![Image 6] -->

<img src="(https://user-images.githubusercontent.com/44580961/100554992-d09c2080-32be-11eb-8a96-c717eabe077b.png)" width="450" height="300" />

Fig. 6 : Creation of Solution	
	
### Step 2: Create the class library for the algorithm 
	
When solution (SumAlgorithm.sln) is created, by default a class library is also created automatically (.cs file).

We have to change the names accordingly. Here for example, change the class library name as **_“Sum.cs”_** as shown in Fig. 6.

_Sum.cs_ serves as the main class folder for the algorithm.

<!--![Image 7]-->

<img src="(https://user-images.githubusercontent.com/44580961/100554994-d265e400-32be-11eb-95b8-dbeaa8f5c69a.png)" width="450" height="300" />

Fig. 7 : The project and class library folder structure

### Step 3: Create the Test folder and Test class library for the algorithm 

We should create a Test folder where we can initiate the program and command the directions. 

	Select the project folder --> Right click --> Add --> New Project

![Image 8](https://user-images.githubusercontent.com/44580961/100554995-d560d480-32be-11eb-8735-1cece51fb26a.png) 

Select the Test file _'MSTest project c#'_ and click on NEXT button as shown in the below Fig. 9.

![Image 9](https://user-images.githubusercontent.com/44580961/100554997-d72a9800-32be-11eb-9991-21e44062166f.png) 

Name the project name as _**SumAlgorithmTest**_ and click on NEXT button. 

![Image 10](https://user-images.githubusercontent.com/44580961/100554998-d98cf200-32be-11eb-9f28-8f19072c1add.png) 

Test project is created under the main solution and rename the class file as _**SumAlgorithmTest1**_ as shown in the below Fig. 11.

<!--![Image 11] -->

<img src="(https://user-images.githubusercontent.com/44580961/100555000-dd207900-32be-11eb-98c7-0db218463908.png)" width="450" height="300" />


### Step 4 : Add NuGet Package 'LearningApi' to both projects 

We should add NuGet package called _LearningApi_ to both project by following the steps below, 

		
	Right click on project (HelloLearningApiExampleAlgorithm/HelloLearningApiExampleAlgorithmTest) --> Click on ‘Manage NuGet packages..’ (Fig. 12)	

	in the pop up window --> Click on BROWSE, (Fig. 13)
	
	search for LearningApi and select --> Select the checkbox of LearningApi nuget --> Click on SELECT/ADD PACKAGE button (Fig. 14)

	
<!--![Image 12]()-->

<img src="https://user-images.githubusercontent.com/44580961/100555001-de51a600-32be-11eb-96a8-6d4b0a446a0b" width="400" height="550" />

Fig. 12 : NuGet package integration step1,

In the pop up, search for the package LearningAPI , select the latest version and click on ADD PACKAGE button.

<!--![Image 13]()-->

<img src="https://user-images.githubusercontent.com/44580961/100555004-e0b40000-32be-11eb-9aef-245dc985f877.png" width="800" height="450" />

Fig. 13 : NuGet package integration step2,  

A pop up with the packages installed along with the LearningApi NuGet package is displayed. Click on OK/Accept button.

### Step 5 : Start the Code for the project and test .cs files  

<a href="#Example_Algoirthm">Click here to recap the LearningApi Example Algorithm</a>

**In Test.cs file** , we direct the model to read the csv file and take the data for training of the model. We also provide data mapper to extract the data from the columns with the following code :

```csharp
api.UseCsvDataProvider(trainDataPath, ',', false);
api.UseDefaultDataMapper();
```
csv data file path is recognised by the line of code :

```csharp
string trainDataPathString = @"SampleData\power_consumption_train.csv";
```
load the meta data (where the features are explained) by the following bit of code:

```csharp
private DataDescriptor LoadMetaData()
        {
            var des = new DataDescriptor();

            des.Features = new Column[3];
            des.Features[0] = new Column { Id = 1, Name = "solar", Index = 0, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
            des.Features[1] = new Column { Id = 2, Name = "wind", Index = 1, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
            des.Features[2] = new Column { Id = 3, Name = "power", Index = 2, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };

            des.LabelIndex = 2;
            return des;
        }
```

In **Algorithm.cs** file , we implement the *IAlgorithm* in the code which is taken from LearningApi NuGet package. *IAlgorithm*  is in the library and it has a separate structure which we have to use in the project as we already have discussed in the section <a href="#LearningApi_Concept">LearningApi Concept</a>. 

![Image 14](https://user-images.githubusercontent.com/44580961/100555006-e3aef080-32be-11eb-9da6-41d0a5089027.png)

Fig. 14 : IAlgorithm interface integrated in the project

Here, In _'IScore Run'_ method, we direct the model to TRAIN (IScore Train) interface where the logic for SUM algorithm is defined. 

The following code is used for training the model :

```csharp
public IScore Train(double[][] data, IContext ctx)
        {
            for (int trainDataIndex = 0; trainDataIndex < numTrainData; trainDataIndex++)
            {
                estimatedOutputLabels[trainDataIndex] = ComputeSum(inputFeatures[trainDataIndex]);
                squareErrors[trainDataIndex] = ComputeSquareError(actualOutputLabels[trainDataIndex], estimatedOutputLabels[trainDataIndex]);
            }

            double meanSquareError = squareErrors.Sum() / numTrainData;

            loss = meanSquareError;

            if (ctx.Score as SumScore == null)
                ctx.Score = new SumScore();

            SumScore scr = ctx.Score as SumScore;
            scr.Loss = loss;

            return ctx.Score;
        }
```
In PREDICT interface, all the logics for computing mean square error is provided as shown in the below code lines. 

```csharp
public IResult Predict(double[][] data, IContext ctx)
        {
            var testData = data;

            int numTestData = testData.Length;

            int numFeatures = ctx.DataDescriptor.Features.Length - 1;

            double[][] inputFeatures = GetInputFeaturesFromData(testData, numFeatures);

            double[] predictedOutputLabels = new double[numTestData];

            for (int testDataIndex = 0; testDataIndex < numTestData; testDataIndex++)
            {
                predictedOutputLabels[testDataIndex] = ComputeSum(inputFeatures[testDataIndex]);
            }

            SumResult res = new SumResult();
            res.PredictedValues = predictedOutputLabels;

            return res;
        }
````

### Step 6 : Create the *Extension.cs* , *Result.cs* and *Score.cs* files

Extension file in a project facilitates other users to utilise our project code in their implementations. Calling this file in other projects enables the project code in other projects.
      
      Right Click on Project name --> Add --> New Class (Fig. 12_left side)
      
      Select Empty class --> Give the class name 'SumExtension' --> Click on NEW button (Fig. right side)

![Image 15](https://user-images.githubusercontent.com/44580961/100555008-e578b400-32be-11eb-99cf-ea1176ff7269.png)

Fig. 15 : Adding Extension class to ALgorithm project 

The following is given as code for extension.cs file in order to use it anywhere in the project further:

```csharp
public static LearningApi UseSum(this LearningApi api)
        {
            var alg = new Sum();
            api.AddModule(alg, "Sum");
            return api;
        }
```

Likewise, in the example solution, the *SumResult.cs* and *SumScore.cs* files should be created to define the values which should be storing the result and trained score data. Follow the steps explained above in Fig.12 to create these classes also.

The values are get and set in the _Result.cs_ file with the following code line :

```csharp
public class SumResult : IResult
    {
        public double[] PredictedValues { get; set; }
    }
```
The values for the Loss are get and set in the _Score.cs_ file with the following lines of code :

```csharp
 public class SumScore : IScore
    {
        public double Loss { get; set; }
    }
 ```

### Step 7 : Result 

According to the algorithm, the set of data of house details is given and trained the model with these data. The data for house price is used to calculate the mean square error and When this score is multiplied with each data given, we get the house price value predicted.

![Image 16]()

Fig. 16 : Result is shown here

***Implement a button - Button name - Basic SUM algorithm in Github ***

<button>Basic SUM algorithm in Github</button>
<a href="#" class="button">Basic SUM algorithm in Github</a>
<button class="button">Basic SUM algorithm in Github</button>

You can refer this example project in the [Click here to refer the SUM algorithm project code in GitHub..](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Individual%20project_AnushaAshokReddy/SumAlgorithm)

-------------------------------------------------------------------------------------------------------------
 ***Implement a button - Button name - Linear Regression Algorithm using LearningApi ***

<button>Linear Regression Algorithm using LearningApi</button>
<a href="#" class="button">Linear Regression Algorithm using LearningApi</a>
<button class="button">Linear Regression Algorithm using LearningApi</button>

# How can you contribute to LearningApi? <a id="Your_Contribution"></a>

If you have implemented a custom module or algorithm and want to integrate it to LearningAPI, then you can do the following, 

- Contact us - implement a page for this 
- Implement your algorithm or/and module
- Create the pull request
- Create an issue in the Repository


