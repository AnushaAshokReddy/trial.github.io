[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/blob/master/LICENSE)

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

Learning API is a Machine Learning Foundation of a set of ML algorithms implemented in .NET Core/C#. It provides a unique pipeline processing API for Machine Learning solutions. Because it is implemented fully in .NET, developers do not have to bridge .NET and Python or other popular ML frameworks. It has been developed in cooperation with daenet GmbH and Frankfurt University of Applied Sciences.

![Image 1](https://user-images.githubusercontent.com/44580961/98464210-a5dc1200-21c1-11eb-95ef-e1a0d7942382.png)

Fig. 1 : daenet GmbH and Frankfurt University of Applied Sciences

LearningAPI already has interfaces that are pre declared which we can easily access, understand and use in our project.

Before you start with the **LearningAPI**, you should get familiarised with several interfaces: IPipeline Module, IAlgorithm, IScore, IResult. These interfaces are shared across all algorithms inside of the **LearningAPI**.

LearningAPI is a foundation of Machine Learning algorithms, which can run in the pipeline of modules compatible to each other. This concept allows using of conceptually different algorithms in the same API, which consists of a chain of modules. Typically in Machine Learning applications, developers need to combine multiple algorithms or tasks to achieve a final task or result.

For example, imagine you want to train a supervised algorithm from historical power consumption data to be able to predict the power consumtion. The training data is contained in the CSV file, which contains features like power consumtion in W, outside temperature, the wind etc. To solve the problem, you first have to read the data from CSV, then to normalize features (ref to normalization todo) and then to train the algorithm.

You could think about these tasks as follows:

1. Read CSV
2. Normalize the data
3. Train the data.

After the above process, you will have a trained instance of the algorithm *algInst*, which can be used for prediction:

4. Use *algInst* to predict the power consumption based on the given temperature and wind.

To solve this problem with the **LearningAPI** the following pseudo can be used:

```
var api = new LearningApi(config)
api.UseCsvDataProvider('csvFileName.csv')
api.UseNormilizerModule();
api.Train();
// Prediction
var predictedPower = api.Predict(108W, 45 wind force);
```
One pipeline module is defined as implementation of interface IPipeline.

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
-------------------------------


An example is explained in the below section -
<a href="#Example_Custom_Algorithm">'PLease refer this section to understand 'How to build a LearningAPI algorithm?'</a>

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
| Euclidian color filter - Deepali | [Github_EuclidianColorFilter Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter) | Available | [Github_EuclidianColorFilter Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/EuclideanColorFilter/Documentation) |
| Image Binarizer | [Github_ImageBinarizer Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer) | Available  | [Github_ImageBinarizer Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/ImageBinarizer/Documentation) |
| Center Module | [Github_CenterModule Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module) | Available | [Github_CenterModule Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/Center%20Module/Documentation) |
| Canny edge detector | [Github_CannyEdgeDetector Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/CannyEdgeDetector) | Available    | Not Available yet |
| SDR Classifier | [Github_SDR Classifier Algorithm](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier) | Available | [Github_SDR Classifier Documentation](https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/tree/master/LearningApi/src/Modules/SDR%20Classifier/Documentation) |

# The LearningApi Concept <a id="LearningApi_Concept"></a>

LearningAPI is a framework for developing software machine learning applications. This includes predefined classes and functions that can be used to process input, train the system and give an accurate predicted answer.

  In order to use LearningApi, we should install the Nuget package called **_LearningApi_** into our project (this will be demonstarted in <a href="#Example_Custom_Algorithm">Example custom algorithm section</a>)
  
  Basically a NuGet package is a single ZIP file with the *.nupkg* extension that contains compiled code (DLLs), other files related to that code, and a descriptive manifest that includes information like the package's version number.
  
  Initially open the class ‘.cs’ and implement the IAlgorithm in the code which is taken from Learning Api NuGet package. IAlgorithm is in the library and it has a separate structure which we have to use in the project. 
  
More information can be found on [Click here for more information on NuGet packages..](https://docs.microsoft.com/en-us/visualstudio/mac/nuget-walkthrough?view=vsmac-2019)

<!--To find out more details, click on [Information..](https://docs.microsoft.com/en-us/nuget/what-is-nuget)-->

**IAlgorithm** - The _IAlgorithm_ interface has 2 phases:

1. _**IResult**_ – IResult is used to set and get the final result of the algorithm and store it. We use IResult for the PREDICT phase - This is the final phase where we get the perfect output for the input provided by the user on the basis of the algorithm we give.In this prediction logic should be written as shown in  screenshot 6.

2. _**IScore**_ – Iscore is used to set and get the values of the variables used in the project. We use IScore for RUN and TRAIN methods.

**RUN**/**TRAIN** – This is the training (learning) part where the random data will be given to our system to test whether the correct output is being displayed after the training session. Here, we call TRAIN method internally.

**TRAIN** – Here we will train the system with our specific set of data I.e input and the output as in how to function. Algorithm of the function is written in Train method.

  **Inputs** to the TRAIN i.e to the algorithm is the set of data with expected outputs for few number of inputs, we train the system and then expect the predicted value to be accurate when other input is given.
  
  **Output** is the predicted value from PREDICT method which gives the accuracy of the True or False statements.
  
**The Pipeline module** receives an input TIN and context information. Usually TIN is set of data, which results as output of th eprevious module. Typically, first module in the pipeline is responsibe to provide learning data and last module in the pipeline is usually algorithm.

# What is a LearningAPI Pipeline Module? <a id="What_is_Module"></a>

A module in Machine Learning represents a set of code that can run independently and perform a machine learning task, given the required inputs. A module might contain a particular algorithm, or perform a task that is important in machine learning, such as missing value replacement, or statistical analysis.
Both algorithms and modules are independent of each other. 

While implementing an algorithm, it is initially trained using various number of data available already to make the algorithm learn how to predict the results for an unknown input in the later stages. Thus the set of data is very important. This data is supposed to be clean with all details. Sometimes in algorithms when we don't get clean data, pipeline modules are used for pre-processing of the data. 

For example some pipeline modules as MinMaxNormalisers have the function of normalising the data for the larger algorithms.

Following example illustrates how to setup the learning pipeline for Data Descriptor:

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
The code shown above setups the pipeline of two modules. 

1.First one is so called action module, which defines the custom code to be executed. 
```csharp
api.UseActionModule<object, double[][]>((notUsed, ctx)
```

2.Second module injects the perceptron algorithm in the pipeline and it is setup by the following line of code:

```csharp
api.UsePerceptron(0.02, 10000);
```

Execution of the pipeline is started with following line of code:

```csharp
IScore score = api.Run() as IScore;
```

When the pipeline starts, modules are executed in the sequence ordered as they are added to the pipeline. 
In this case, first action module will be executed and then perceptron algorithm. After running of the pipeline, model is trained. Next common step in Machine Learning applications is called evaluation of the model. Following code in previous example shows how to evaluate (predict) the model:

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

# What is Algorithm? <a id="What_is_Algorithm"></a>

Machine learning is a class of methods for automatically creating models from data. Machine learning algorithms are the engines of machine learning, meaning it is the algorithms that turn a data set into a model. Which kind of algorithm works best (supervised, unsupervised, classification, regression, etc.) depends on the kind of problem you’re solving, the computing resources available, and the nature of the data.

An algorithm is a set of logical coding which is trained with lots and lots of data to predict the otput most accurately.

# How to build the custom algorithm using LearningAPI? <a id="Example_Custom_Algorithm"></a>

  The below solution demonstrates how to implement a Linear Regression algorithm using LearningAPI. To understand the implementation, you should initially understand the linear Regression concept. 
	
## LearningApi Example Algorithm <a id="#Example_Algoirthm"></a>

Lets take a simple example to build a model to predict the HOUSE PRICE. For this, we have 3 phases of tasks to do. 

1. Input Phase
2. Training Phase
3. Output 

**Input phase :** 

Let's consider 2 simple features (features are the inputs)
	
	*Size* - Size of the house
	
	*Room* - Number of rooms
	
and 1 label (the output features is called label)

	*Price* - Price of the house based on *room* feature
	
Now we should train the model with the above 2 features and a label by giving a real data (shown below) in csv file, we have taken the normalised data  :

| Size | Room | Price | 
|:--- |:--- |:--- |
| 1 | 1 | 6 |
| 1 | 2 | 8 |
| 2 | 1 | 9 |
| 2 | 2 | 11 |
| 3 | 1 | 12 |
| 3 | 2 | 14 |
| 4 | 1 | 15 |
| 4 | 2 | 17 |
| 5 | 1 | 18 |
| 5 | 2 | 20 |

The model reads the csv file with the following interface :

```csharp
 api.UseCsvDataProvider(trainDataPath, ',', false);
```
we should provide the meta data and label index for our csv file in the datadescriptor. Meta data is the declaration of columns in csv file and label index is the index number of feature based on whihc the prediction should be made. 

The following is the code snippet for the datadescriptor :

```csharp
 private DataDescriptor LoadMetaData()
        {
            var des = new DataDescriptor();

            des.Features = new Column[3];

            des.LabelIndex = 2;
            return des;
        }
```

**Training phase :** The model reads the data in from csv file and based on the data given, after training the model, the model should be able to predict the result y. 

The linear regression formula for the above situation is **PRICE = 3 * SIZE + 2 * ROOM + 1 **

This is similar to **y = m1 x1 + m2 x2 + c** , 

where y is the predicted price of the house

      x1 is the Size of house based on which the price should be predicted
      
      x2 is the Room 
      
      m1/3 is the weight (w1) for feature PRICE and m2/2 i the weight (w2) for feature ROOM
      
      c/1 is the bias (b)
      
Here m and c are weight and bias - Suppose the model has to take a predict the price of the house between many prices. So it will choose one value after analyzing all different options. Analyzing will give some percentage prediction to all the values selected by 'w' on the basis of experience. This percentage is weight in terms of ML. These percentages do not have to be exact always. It might be wrong and it will be confirmed after crossing the values. Whatever will be the value selected by weight accordingly percentages will change , so that next time prediction accuracy will be more as compared to previos predictions.

we use *Mean Square Error* concept here to find out the error differences and help the model to finalise the least error value to be more accurate for the prediction. 

Initially let's initialise the weight and bias (let this be w1=2 , w2=2 and b=1). This w and b are used for calculating the prices across all the data of size given :

**1st data : w1=2 , w2=2 and b=1, size 10sq.m**

	substitute in formula -- y1^ = 2*1 + 2*1 + 1 = 5
	
	Square Error -- SE1 = (actual price for size 1 - predicted price for size 1)^2 
	
			= (6 - 5)^2 = 1^2 = 1
			
**2nd data : w=2 and b=1, size 20**

	substitute in formula -- y1^ = 2*1 + 2*2 + 1 = 7
	
	Square Error -- SE2 = (actual price for size 1 - predicted price for size 1)^2 
	
			= (8 - 7)^2 = 1^2 = 1
			
this calculation occurs for all the values of size and room features with the initialised w and b values.

Mean square error calculation (MSE1)-- 

    	     = (SE1+SE2+....)/ 10
    
    	     = (1+1*....)/10
    
	MSE1 = x

```csharp
for (int trainDataIndex = 0; trainDataIndex < numTrainData; trainDataIndex++)
                {
                    estimatedOutputLabels[trainDataIndex] = ComputeOutput(inputFeatures[trainDataIndex], weights, bias);
                    squareErrors[trainDataIndex] = ComputeSquareError(actualOutputLabels[trainDataIndex], estimatedOutputLabels[trainDataIndex]);
                }

                double meanSquareError = squareErrors.Sum() / numTrainData;

                loss[epoch] = meanSquareError;
```

Likewise this will continue with other weights and biases. New weights and biases will be calculated by 
- finding the least w and b values by Gradient Descent concept
- finding Slopes of w and b 
- setting up the Leanring rate (LR)

Hence new w and new b values would be :

	W1(new) = W1(old) - (slope derivative of w1 * LR)
	
	W2(new) = W2(old) - (slope derivative of w2 * LR)
	
	b(new) = b(old) - (slope derivative of b * LR)

```csharp
Tuple<double[], double> hyperParameters = GradientDescent(actualOutputLabels, estimatedOutputLabels, inputFeatures, numTrainData, numFeatures);

                // Partial derivatives of loss with respect to weights
                double[] dWeights = hyperParameters.Item1;

                // Updating weights
                for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++)
                {
                    weights[featureIndex] = weights[featureIndex] - m_LearningRate * dWeights[featureIndex];
                }

                // Partial derivative of loss with respect to bias
                double dbias = hyperParameters.Item2;

                // Updating bias
                bias = bias - m_LearningRate * dbias;
            }
```

Likewise the model will do these calculations by assuming several weights and biases.

So, the process from initialising of w and b till finding the W(new) and b(new) values is called 1 epoch. Let's initialise number of epochs for training out model. 

The following lines of code initialises LR and epochs at the starting of the program - 

```csharp
	private double m_LearningRate;
        private int m_Epochs;

        public HelloLearningApiExampleAlgorithm (double learningRate, int epochs)
        {
            m_LearningRate = learningRate;
            m_Epochs = epochs;
        }
```

Now in 1st epoch, MSE1 is some x value, likewise after 1000 epochs, MSE20 would be (lets assume) some 0.001. Therefore the mean square error is completely reduced which means the error is less and the W and B value of this least MSE is the right value for the final formula. 

Assume, for the correct epoch (1000th epoch), model has taken the values of W1=2.98, W2=2.02 and B=1.02 (near to the value W1=3, W2=2 and B=1 as shown in the initial forula of the logic).

	PRICE = 2.98 * SIZE + 2.02 * ROOM + 1.02
	
**Output Phase :** Now, the model has an exact formula (found out as the output of training). This formula will be used by the model to predict price of the house with any further sizes of the house. 

## Implementation of LearningApi for the above example Algorithm :

### Step 1: Create a solution 

In the Visual Studio, create a new solution by following the steps -
	
    Navigate to File --> New --> Project

Use the selectors on the left side to choose the different types of programming languages or platforms to work with. For example, we are creating a class library with the template .NET STANDARD under the Visual C# selector as show in Fig. 4.

    Click on NEXT 	

<!--![Image 4](https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png) -->

<img src="https://user-images.githubusercontent.com/44580961/98464414-04ee5680-21c3-11eb-82fe-910a29ed7d4d.png" width="600" height="450" />

Fig. 4 : New Project

For our example - given the project name as **“HelloLearningApiExampleAlgorithm”**	

    Name the project --> Solution Name --> Specify the location --> Click OK/CREATE
    
<!--![Image 5]() -->

<img src="https://user-images.githubusercontent.com/44580961/99399484-bac84c00-290b-11eb-93b1-504faf36eec1.png" width="600" height="450" />

Fig. 5 : Project and Solution name

Now the project is created with the name _'HelloLearningApiExampleAlgorithm.sln'_
  
<!--![Image 6](https://user-images.githubusercontent.com/44580961/98464421-0ddf2800-21c3-11eb-9951-f66298e25891.png) -->

<img src="(https://user-images.githubusercontent.com/44580961/99438805-5375c080-293a-11eb-928a-ba234162a1ea.png)" width="450" height="300" />

Fig. 6 : Creation of Solution	
	
### Step 2: Create the class library for the algorithm 
	
When solution(HelloLearningApiExampleAlgorithm.sln) is created, by default a class library is also created automatically (.cs file).

We have to change the names accordingly. Here for example, change the class library name as “ExampleLearningApiAlgorithm.cs” as shown in Fig. 6.

ExampleLearningApiAlgorithm.cs serves as the main class folder for the algorithm.

<!--![Image 7]-->

<img src="(https://user-images.githubusercontent.com/44580961/99438878-743e1600-293a-11eb-985d-20de14d1dda5.png)" width="450" height="300" />

Fig. 7 : The project and class library folder structure

### Step 3: Create the Test folder and Test class library for the algorithm 

We should create a Test folder where we can initiate the program and command the directions. 

	Select the project folder --> Right click --> Add --> New Project

![Image 8](https://user-images.githubusercontent.com/44580961/99399514-c4ea4a80-290b-11eb-939f-4ea14c0ee485.png) 

Select the Test file 'MSTest project c#' and click on NEXT button as shown in the below Fig. 9.

![Image 9](https://user-images.githubusercontent.com/44580961/99399521-c87dd180-290b-11eb-8e5b-42adaa90a054.png) 

Name the project name as **HelloLearningApiExampleAlgorithmTest** and click on NEXT button. 

![Image 10](https://user-images.githubusercontent.com/44580961/99399539-d16ea300-290b-11eb-9336-5d3f6710190b.png) 

Test project is created under the main solution and rename the class file as HelloLearningApiExampleAlgorithmTest1 as shown in the below Fig. 11.

<!--![Image 11](https://user-images.githubusercontent.com/44580961/99399545-d3d0fd00-290b-11eb-9c1a-135f301f7f64.png) -->

<img src="(https://user-images.githubusercontent.com/44580961/99399545-d3d0fd00-290b-11eb-9c1a-135f301f7f64.png)" width="450" height="300" />


### Step 4 : Add NuGet Package 'LearningApi' to both projects 

We should add NuGet package called _LearningApi_ to both project by following the steps below, 

		
	Right click on project (HelloLearningApiExampleAlgorithm/HelloLearningApiExampleAlgorithmTest) --> Click on ‘Manage NuGet packages..’ (Fig. 12)	

	in the pop up window --> Click on BROWSE, (Fig. 13)
	
	search for LearningApi and select --> Select the checkbox of LearningApi nuget --> Click on SELECT/ADD PACKAGE button (Fig. 14)

	
<!--![Image 12]()-->

<img src="https://user-images.githubusercontent.com/44580961/99399553-d7fd1a80-290b-11eb-8a42-8e6e11eb47a2.png" width="400" height="550" />

Fig. 12 : NuGet package integration step1,

In the pop up, search for the package LearningAPI , select the latest version and click on ADD PACKAGE button.

<!--![Image 13]()-->

<img src="https://user-images.githubusercontent.com/44580961/99399561-daf80b00-290b-11eb-868e-e4ce3329ad56.png" width="800" height="450" />

Fig. 13 : NuGet package integration step2,  

A pop up with the packages installed along with the LearningApi NuGet package is displayed. Click on OK/Accept button.

### Step 5 : Start the Code for the project and test .cs files  <a href="#Example_Algoirthm">LearningApi Example Algorithm</a>

**In Test.cs file** , we direct the model to read the csv file and take the data for training of the model. We also provide data mapper to extract the data from the columns with the following code :

```csharp
api.UseCsvDataProvider(trainDataPath, ',', false);
api.UseDefaultDataMapper();
```
csv data file path is recognised by the line of code :

```csharp
string testDataPathString = @"SampleData\house_price_train.csv";
```
load the meta data (where the features are explained) by the following bit of code:

```csharp
private DataDescriptor LoadMetaData()
        {
            var des = new DataDescriptor();

            des.Features = new Column[3];
            des.Features[0] = new Column { Id = 1, Name = "size", Index = 0, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
            des.Features[1] = new Column { Id = 2, Name = "rooms", Index = 1, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };
            des.Features[2] = new Column { Id = 3, Name = "price", Index = 2, Type = ColumnType.NUMERIC, DefaultMissingValue = 0, Values = null };

            des.LabelIndex = 2;
            return des;
        }
```

**In Algorithm.cs file** , we implement the *IAlgorithm* in the code which is taken from LearningApi NuGet package. *IAlgorithm*  is in the library and it has a separate structure which we have to use in the project as we already have discussed in the section <a href="#LearningApi_Concept">LearningApi Concept</a>. 

![Image 14](https://user-images.githubusercontent.com/44580961/99401608-62467e00-290e-11eb-8197-6dfa9aa06f32.png)

Fig. 14 : IAlgorithm interface integrated in the project

Here, In _'IScore Run'_ we direct the model to TRAIN interface where the logic for algorithm is defined. 

The following code is used for training the model :

```csharp
for (int epoch = 0; epoch < m_Epochs; epoch++)
            {
                for (int trainDataIndex = 0; trainDataIndex < numTrainData; trainDataIndex++)
                {
                    estimatedOutputLabels[trainDataIndex] = ComputeOutput(inputFeatures[trainDataIndex], weights, bias);
                    squareErrors[trainDataIndex] = ComputeSquareError(actualOutputLabels[trainDataIndex], estimatedOutputLabels[trainDataIndex]);
                }

                double meanSquareError = squareErrors.Sum() / numTrainData;

                loss[epoch] = meanSquareError;
                
                Tuple<double[], double> hyperParameters = GradientDescent(actualOutputLabels, estimatedOutputLabels, inputFeatures, numTrainData, numFeatures);

                // Partial derivatives of loss with respect to weights
                double[] dWeights = hyperParameters.Item1;

                // Updating weights
                for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++)
                {
                    weights[featureIndex] = weights[featureIndex] - m_LearningRate * dWeights[featureIndex];
                }

                // Partial derivative of loss with respect to bias
                double dbias = hyperParameters.Item2;

                // Updating bias
                bias = bias - m_LearningRate * dbias;
            }

            if (ctx.Score as LinearRegressionScore == null)
                ctx.Score = new LinearRegressionScore();

            LinearRegressionScore scr = ctx.Score as LinearRegressionScore;
            scr.Weights = weights;
            scr.Bias = bias;
            scr.Loss = loss;            

            return ctx.Score;

```
In PREDICT interface, all the logics for computing mean square error, outputlabels are provided. 

### Step 6 : Create the *Extension.cs* , *Result.cs* and *Score.cs* files

Extension file in a project facilitates other users to utilise our project code in their implementations. Calling this file in other projects enables the project code in other projects.
      
      Right Click on Project name --> Add --> New Class (Fig. 12_left side)
      
      Select Empty class --> Give the class name ExampleLearningApiAlgorithmExtension --> Click on NEW button (Fig. right side)

![Image 15](https://user-images.githubusercontent.com/44580961/99399573-df242880-290b-11eb-8487-36d3aeaf4b28.png)

Fig. 15 : Adding Extension class to ALgorithm project 

The following is given as code for extension.cs file in order to use it anywhere in the project further:

```csharp
public static LearningApi UseExampleLearningApiAlgorithm(this LearningApi api, double learningRate, int epochs)

        {
            var alg = new ExampleLearningApiAlgorithm(learningRate, epochs);
            api.AddModule(alg, "Linear Regression");
            return api;
        }
```

Likewise, in the example solution, the *ExampleLearningApiAlgorithmResult.cs* and *LearningApiAlgorithmScore.cs* files should be created to define the values which should be storing the result and trained score data. Follow the steps explained above in Fig.12 to create these classes also.

The values are get and set in the _Result.cs_ file with the following code line :

```csharp
public class ExampleLearningApiAlgorithmResult : IResult
    {
        public double[] PredictedValues { get; set; }
    }
```
The values for the features are get and set in the _Score.cs_ file with the following lines of code :

```csharp
public class LinearRegressionScore : IScore
    {
        public double[] Weights { get; set; }

        public double Bias { get; set; }

        public double[] Loss { get; set; }
    }
 ```

### Step 7 : Result 

According to the algorithm, the set of data of house details is given and trained the model with these data. The data for house price is used to calculate the mean square error and When this score is multiplied with each data given, we get the house price value predicted.

![Image 16]()

Fig. 16 : Result is shown here

You can refer this example project in the [Example algorithm project in GitHub..](https://github.com/UniversityOfAppliedSciencesFrankfurt/se-dystsys-2018-2019-softwareengineering/tree/Anusha_Ashok_Reddy/My%20work/My%20Project)

-------------------------------------------------------------------------------------------------------------
 

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

# How can you contribute to LearningApi? <a id="Your_Contribution"></a>

If you have implemented a custom module or algorithm and want to integrate it to LearningAPI, then you can do the following, 

- Contact us - implement a page for this 
- Implement your algorithm or/and module
- Create the pull request
- Create an issue in the Repository


