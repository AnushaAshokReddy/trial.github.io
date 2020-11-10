# Welcome To LearningApi Tutorial 

# Index 

<a href="#LearningApi_Introduction">LearningApi Introduction</a>

<a href="#LearningApi_Concept">LearningApi Concept</a>

<a href="#What_is_Algorithm">What is Algorithm?</a>

<a href="#Example_Custom_Algorithm">How to build the custom algorithm?</a>

<a href="#What_is_Module">What is Module?</a>

<a href="#Example_Custom_Module">How to build the custom module?</a>

<a href="#Supported_Algorithms&Modules_List">Supported Algorithms and Modules</a>

<a href="#Your_Contribution">Contribution to Learning API?</a>


# LearningApi Introduction <a id="LearningApi_Introduction"></a>

Learning API is Machine Learning Foundation of a set of ML libraries fully implemented as .NET Standard library. It provides a unique processing API for Machine Learning solutions. Because it is implemented fully in .NET, developers do not have to bridge .NET and Python or other popular ML frameworks. It has been developed in cooperation with Daenet GmbH and Frankfurt University of Applied Sciences.

![Image 1](https://user-images.githubusercontent.com/44580961/98464210-a5dc1200-21c1-11eb-95ef-e1a0d7942382.png)

Fig. 1 : Daenet GmbH and Frankfurt University of Applied Sciences

LearningAPI already has interfaces pre declared which we can easily access, understand and use in our project.

For example IAlgorithm, IScore, IResult, IPipeline module.

<!--![Image 2](https://user-images.githubusercontent.com/44580961/98464406-fb64ee80-21c2-11eb-9dc1-3fcb08e1d0fc.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464406-fb64ee80-21c2-11eb-9dc1-3fcb08e1d0fc.png" width="700" height="200" />

An example code is shown in the Fig. 3 : 

<!--![Image 3](https://user-images.githubusercontent.com/44580961/98464411-01f36600-21c3-11eb-877f-3c3a3815b1c0.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464411-01f36600-21c3-11eb-877f-3c3a3815b1c0.png" width="700" height="300" />

LearningAPI is a foundation of Machine Learning algorithms, which can run in the pipeline of modules compatible to each other. This concept allows using of conceptually different algorithms in the same API, which consists of a chain of modules. One pipeline module is defined as implementation of interface IPipeline. 

The structure for IPipeline Interface: 
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

LearningAPI is a framework for developing software machine learning applications. This includes predefined classes and functions that can be used to process input, train the system and give an accurate predicted answer.

  In order to use LearningApi, we should install the Nuget package called **_LearningApi_** into our project (this will be demonstarted in <a href="#Example_Custom_Algorithm">Example custom algorithm section</a>
  
  Basically a NuGet package is a single ZIP file with the *.nupkg* extension that contains compiled code (DLLs), other files related to that code, and a descriptive manifest that includes information like the package's version number.
  
  Initially open the class ‘.cs’ and implement the IAlgorithm in the code which is taken from Learning Api NuGet package. IAlgorithm is in the library and it has a separate structure which we have to use in the project. 
  
More information can be found on [Click here for more information on NuGet packages..](https://docs.microsoft.com/en-us/visualstudio/mac/nuget-walkthrough?view=vsmac-2019)

To find out more details, click on [Information..](https://docs.microsoft.com/en-us/nuget/what-is-nuget)

**IAlgorithm** - The _IAlgorithm_ interface has 2 phases:

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

# What is Algorithm? <a id="What_is_Algorithm"></a>

Machine learning is a class of methods for automatically creating models from data. Machine learning algorithms are the engines of machine learning, meaning it is the algorithms that turn a data set into a model. Which kind of algorithm works best (supervised, unsupervised, classification, regression, etc.) depends on the kind of problem you’re solving, the computing resources available, and the nature of the data.

An algorithm is a set of logical coding which is trained with lots and lots of data to predict the otput most accurately.

# How to build the custom algorithm? <a id="Example_Custom_Algorithm"></a>

  The below solution demonstrates how to implement a custom algorithm. In this example, the SUM and AVERAGE logics will be impemented.
  
  This example is only for reference on steps to implement a solution using LearningApi. 
	
## LearningApi Example Algorithm <a id="#Example_Algoirthm"></a>

Prediction of 'Chance of Precipitation' by calculating the average of temperature data and the average of chance of precipitation till date given. Motive is to achieve the solution using LeaningApi framework.

## Example Solution using LearningApi Algorithm :

### Step 1: Create a solution 

In the Visual Studio, create a new solution by following the steps -
	
    Navigate to File --> New --> Project

Use the selectors on the left side to choose the different types of programming languages or platforms to work with. For example, we are creating a class library with the template .NET STANDARD under the Visual C# selector as show in Fig. 4.

    Click on NEXT 	

<!--![Image 4](https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png" width="600" height="450" />

Fig. 4 : New Project

For our example - given the project name as **“HelloLearningApiAlgorithm”**	

    Name the project --> Solution Name --> Specify the location --> Click OK/CREATE
    
<!--![Image 5](https://user-images.githubusercontent.com/44580961/98464418-0ae43780-21c3-11eb-9d19-9c08e951e4e9.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464418-0ae43780-21c3-11eb-9d19-9c08e951e4e9.png" width="600" height="450" />

Fig. 5 : Project and Solution name

Now the project is created with the name _'HelloLearningApiAlgorithm.sln'_
  
<!--![Image 6](https://user-images.githubusercontent.com/44580961/98464421-0ddf2800-21c3-11eb-9951-f66298e25891.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464421-0ddf2800-21c3-11eb-9951-f66298e25891.png" width="550" height="300" />

Fig. 6 : Creation of Solution	
	
### Step 2: Create the class library for the algorithm 
	
When solution(HelloLearningApiAlgorithm.sln) is created, by default a class library is also created automatically (.cs file).

We have to change the names accordingly. Here for example, change the class library name as “LearningApiAlgorithm.cs” as shown in Fig. 6.

LearningApiAlgorithm.cs serves as the main class folder for the algorithm.

![Image 7](https://user-images.githubusercontent.com/44580961/98464425-16cff980-21c3-11eb-92ca-26aee694db54.png) 

Fig. 7 : The project and class library folder structure
	
### Step 3 : Add NuGet Package 'LearningApi' to our project 

We should add NuGet package called _LearningApi_ to our project by following the steps below, 

		
	Right click on project (HelloWorldTutorial.sln) --> Click on ‘Manage NuGet packages..’ (Fig. 8)	

	in the pop up window --> Click on BROWSE, (Fig. 9)
	
	search for LearningApi and select --> Select the checkbox of LearningApi nuget --> Click on SELECT/ADD PACKAGE button (Fig. 10)

	
<!--![Image 8](https://user-images.githubusercontent.com/44580961/98464428-1a638080-21c3-11eb-9789-9788f5e01a95.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464428-1a638080-21c3-11eb-9789-9788f5e01a95.png" width="400" height="550" />

Fig. 8 : NuGet package integration step1,

<!--![Image 9](https://user-images.githubusercontent.com/44580961/98464429-1df70780-21c3-11eb-9e40-7393ae09c9b8.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464429-1df70780-21c3-11eb-9e40-7393ae09c9b8.png" width="600" height="450" />

Fig. 9 : NuGet package integration step2,  

<!--![Image 10](https://user-images.githubusercontent.com/44580961/98464431-218a8e80-21c3-11eb-8329-be2fe49b26e3.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464431-218a8e80-21c3-11eb-8329-be2fe49b26e3.png" width="600" height="450" />

Fig. 10 : NuGet package integration step3
  
A pop up with the packages installed along with the LearningApi NuGet package is displayed. Click on OK button.

### Step 4 : Start the Code for the project <a href="#Example_Algoirthm">LearningApi Example Algorithm</a>

Open the class *‘LearningApiAlgorithm.cs’* and implement the *IAlgorithm* in the code which is taken from LearningApi NuGet package. *IAlgorithm*  is in the library and it has a separate structure which we have to use in the project as we already have discussed in the section <a href="#LearningApi_Concept">LearningApi Concept</a>. 

![Image 11](https://user-images.githubusercontent.com/44580961/98464507-ad041f80-21c3-11eb-8ad0-4f8403371402.png)

Fig. 11 : IAlgorithm interface integrated the project

### Step 5 : Create the *Extension.cs* , *Result.cs* and *Score.cs* files

Extension file in a project facilitates other users to utilise our project code in their implementations. Calling this file in other projects enables the project code in other projects.
      
      Right Click on Project name --> Add --> New Class (Fig. 12_left side)
      
      Select Empty class --> Give the class name --> Click on NEW button (Fig. right side)

![Image 12](https://user-images.githubusercontent.com/44580961/98464510-b3929700-21c3-11eb-8fe0-3df830c1f65f.png)

Fig. 12 : Adding new class 

Likewise, in the example solution, the *LearningApiAlgorithmResult.cs* and *LearningApiAlgorithmScore.cs* files should be created to define the values which should be storing the result and trained score data. Follow the steps explained above in Fig.12 to create these classes also.

In the example solution, these 3 classes are created to demonstarte the structure of the complete solution as shown in Fig. 13.

<!--![Image 13](https://user-images.githubusercontent.com/44580961/98464512-b68d8780-21c3-11eb-816f-9a5015f14b43.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464512-b68d8780-21c3-11eb-816f-9a5015f14b43.png" width="550" height="300" />

Fig. 13 : Complete solution model

### Step 6 : Coding part for the example algorithm.   

We give the main algorithm in the _LearningApiAlgorithm.cs_ file under TRAIN module as shown below,

<!--![Image 14](https://user-images.githubusercontent.com/44580961/98464513-b8574b00-21c3-11eb-8f82-3d9ac614c908.png)-->

<img src="https://user-images.githubusercontent.com/44580961/98464513-b8574b00-21c3-11eb-8f82-3d9ac614c908.png" width="500" height="400" />

Fig. 14 : Learning API Algorithm code implementation

### Step 7 : Result 

According to the algorithm, the set of data of temperature is given and taken the average of the temperature. The data for chance of precipitation is taken an average. The ratio of average of temperature and average of chance of precipitation is given to be our score. When this score is multiplied with each data given, we get the precipitation value predicted.

![Image 15]()

Fig. 15 : Result is shown here

You can refer this example project in the [Example algorithm project in GitHub..](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-dystsys-2018-2019-softwareengineering/tree/Anusha_Ashok_Reddy/My%20work/My%20Project)

-------------------------------------------------------------------------------------------------------------
# What is Module? <a id="What_is_Module"></a>

A module in Machine Learning represents a set of code that can run independently and perform a machine learning task, given the required inputs. A module might contain a particular algorithm, or perform a task that is important in machine learning, such as missing value replacement, or statistical analysis.
Both algorithms and modules are independent of each other. 

While implementing an algorithm, it is initially trained using various number of data available already to make the algorithm learn how to predict the results for an unknown input in the later stages. Thus the set of data is very important. This data is supposed to be clean with all details. Sometimes in algorithms when we don't get clean data, pipeline modules are used for pre-processing of the data. 

For example some pipeline modules as MinMaxNormalisers have the function of normalising the data for the larger algorithms. 

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
| Euclidian color filter - Deepali | [Github_EuclidianColorFilter Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter) | Available | [Github_EuclidianColorFilter Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter/Documentation) |
| Image Binarizer | [Github_ImageBinarizer Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer) | Available  | [Github_ImageBinarizer Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer/Documentation) |
| Center Module | [Github_CenterModule Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module) | Available | [Github_CenterModule Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module/Documentation) |
| Canny edge detector | [Github_CannyEdgeDetector Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/CannyEdgeDetector) | Available    | Not Available yet |
| SDR Classifier | [Github_SDR Classifier Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier) | Available | [Github_SDR Classifier Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier/Documentation) |

# How can you contribute to LearningApi? <a id="Your_Contribution"></a>

If you have implemented a custom module or algorithm and want to integrate it to LearningAPI, then you can do the following, 

- Contact us - implement a page for this 
- Implement your algorithm or/and module
- Create the pull request
- Create an issue in the Repository


