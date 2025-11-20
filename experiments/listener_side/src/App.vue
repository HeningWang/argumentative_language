<template>
  <Experiment 
  	title='Utterance choice' 
  >
    <Screen :title="'Intro'">
		<Slide>
			Thank you for your participation in our study!
			Your anonymous data makes an important contribution to our understanding of human language use.
			<br />
			<br />
			Legal information:
			Before starting the experiment, please read the legal information on the next page carefully.
			<br />
			<br />
			You must be at least 18 years old to participate.
			<br />
			Your anonymity is assured; the researchers who have requested your 
			participation will not receive any personal information about you.
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
			
		<Slide>
			<b>Study title</b>: Describing exam results
			<br /><br />
			<b>Principal Investigator</b>: Michael Franke
			<br /><br />
			<b>Researcher collecting data</b>: Fausto Carcassi
			<br /><br />
			<b>What is this document?</b>  This document explains what kind of study we’re doing, what your rights are, and what will be done with your data. You should print this page for your records.
			<br /><br />
			<b>Nature of the study.</b> You are invited to participate in a study which involves looking at sets of (fictitious) exam results. You will be asked to describe these results. Once you finish, we will have also some brief questions concerning your age, gender, and language background. Your session should last for up to 10 minutes. You will be given full instructions shortly.
			<br /><br />
			<b>Compensation.</b> You will be paid £1.5 for your participation in this study.
			<br /><br />
			<b>Risks and benefits.</b> There are no known risks to participation in this study. Other than the payment mentioned, there are no tangible benefits to you, however you will be contributing to our knowledge about language.  
			<br /><br />
			<b>Confidentiality and use of data.</b> All the information we collect during the course of the research will be processed in accordance with Data Protection Law. In order to safeguard your privacy, we will never share personal information (like your name) with anyone outside the research team. Your data will be referred to by a unique participant number. Please note that we will temporarily collect your participant ID to prevent repeated participation, however we will never share this information with anyone outside the research team. We will store any personal data (i.e., participant ID) securely using the University of Tübingen’s storage systems. The anonymised data collected during this study will be used for research purposes and may be shared with other researchers or with the general public (e.g., we may make it available through the world wide web, or use it in TV or radio broadcasts).
			<br /><br />
			<b>Voluntary participation and right to withdraw.</b> Your participation is voluntary, and you may withdraw from the study at any time and for any reason. If you withdraw from the study after data gathering, we will delete your Prolific ID from the data and there is no penalty or loss of benefits to which you are otherwise entitled. To withdraw from the study after data gathering please contact the research team by providing your prolific participant ID.  
			<br /><br />
			If you have any questions about what you’ve just read, please feel free to ask, or contact us later. You can contact us by email at fausto.carcassi@gmail.com.
			<br /><br />
			By accepting this task, you consent to the following:
			<br /><br />
			<span style="margin-left:2em">1. <b>I agree to participate in this study.</b></span>
			<br /><br />
			<span style="padding-left:2em">2. I confirm that I have read and understood <b>how my data will be stored and used.</b></span>
			<br /><br />
			<span style="margin-left:2em">3. I understand that I have the <b>right to terminate this session</b> at any point. If I choose to <b>withdraw after completing the study</b> by contacting the research team until <b>31/03/2023</b>, my <b>Prolific ID will be deleted from the data</b> at that time.</span>
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
			Many events in life can be interpreted in one way or another. Think of describing the same glass as either half full or half empty. In this experiment, we are interested in how <i>you</i> would creatively use language to stress one view of a situation or another.
			<br/><br/>
			We will show you pictures of situations (results from an exam). Your task is to write a description of that situation, which might emphasize or deemphasize a certain impression in the mind of a person who has not seen the picture. <strong>Please refrain from giving blatantly false descriptions (no lies!)</strong>.
			<br/><br/>
			The next screen will give you more concrete instructions for the upcoming task.
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
			<strong>Background story:</strong> Please imagine you have been hired as a marketing consultant for Green Valley High School. Part of your job is to write a report on the results of standardized math exam questions. These results have been published for Green Valley and for your main rival, Riverside High School.
			<br/><br/>
			It's important that you don't tell any lies in the report, but you don't have to report objectively on the facts. <strong>Your aim is to make Green Valley sound like a school whose students have a high probability of success on the exam questions, and Riverside sound like a school whose students have a low probability of success.</strong>
			<br/><br/>
			In the following, you will be presented with tables showing the results of students who took a math exam for each high school.
			The tables show the number of questions each student answered correctly or incorrectly. For each correctly answered question, the table contains a "<i style=color:#13AC38>  &#10004 </i>", and for each incorrectly answered question, the table contains a "<i style=color:#B12810> &#10008 </i>".
			<br />
			<br />
			Given the information from the table, please <strong>select a sentence which truthfully describes the situation and which emphasizes a high or low overall probability of success</strong> (as indicated in each trial).
			<br />
			<br />
			The next screen will show you an example.
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
			Here is an example of a picture showing exam results. For each student listed, you see an indication of how many questions that student answered correctly. 
			Remember that for each correctly answered question, 
			the table contains a "<i style=color:#13AC38>  &#10004 </i>", and for each incorrectly answered question, the table contains a "<i style=color:#B12810> &#10008</i>
			<br />
			<br />
			<AnswersTable :trialData='sampleTrial' />
			<br />
			<br />
			If these are results from Green Valley, so that your task is to give the impression that students, on average, have a high probability of success, you could select an utterance like:
			<br />
			"Some of the students got all of the questions right."
			<br />
			<br />
			If these are results from Riverside, so that your task is to make this sound as if students, on average, have a low probability of success, you could say something like:
			<br />
			"Some of the students got none of the questions right."
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
			Great! You've completed the training phase. We will move on to the main part of the experiment next.
			<br />
			<br />
			Remember: the tables are showing the exam results of students of the statistics class.
			A "<i style=color:#13AC38>  &#10004 </i>" indicates that a task was rated as correctly, "<i style=color:#B12810> &#10008 </i>" that it was rated as wrong.
			<br />
			<br />
			Given the information from the table, you should present Riverside's results in such a way as to give the impression that Riverside has a low rate of success,
			and to present Green Valley's results to give the impression that is has a high rate of success</strong>, but avoid telling lies.
			<button @click="$magpie.nextScreen()">Next</button>
		</Slide>
    </Screen>
	
	// After all the instructions,
	// the actual trials!
	<template v-for="(trial, i) in trials">
		<SelectionScreen
			:question="questions[trial.condition]"
			text="%s of the students got %s of the answers %s"
			:options="choices"
			:key='i'
			:progress='i / trials.length'
		>
			// Add table to the stimulus slot in the slide 
			<template #stimulus>
				<AnswersTable :trialData='trial' />
				<Record :data="trial" />
			</template>
		</SelectionScreen>
	</template>

	<PostTestScreen />
	<SubmitResultsScreen />
  </Experiment>
</template>

<script>
import _ from 'lodash';
import AnswersTable from '@/Answers.vue';
import SelectionScreen from '@/SelectionScreen.vue';


function defineSampleTrial(arraySizeCondition){
	// Has to be compatible with 'Some of the students got all of the questions right'
	// and 'Some of the students got all of the questions wrong'
	var sampleTrial;
	if (arraySizeCondition === 'high'){
		sampleTrial = {
			"condition": 1,
			"names": [ "Marie", "Nico", "Mia", "John", "Alex"],
			"studentsArray": [12, 12, 9, 0, 0],
			"nQuestions": 12 
		};
	} else if (arraySizeCondition === 'info'){
		sampleTrial = {
			"condition": 1,
			"names": ["Marie", "Nico", "Mia", "John", "Alex"],
			"studentsArray": [12, 12, 9, 0, 0],
			"nQuestions": 12 
		};
	} else if (arraySizeCondition === 'low'){
		sampleTrial = {
			"condition": 1,
			"names": [ "Marie", "Nico", "Mia", "John", "Alex" ],
			"studentsArray": [ 6, 6, 1, 0, 0 ],
			"nQuestions": 6 
		};
	} 
	}
	console.log(sampleTrial);
	return sampleTrial
}

function constructTrials(arraySizeCondition){
	// Return a list of objects
	// Each object represents a trial and contains
	// information about:
		// condition (high or low, represented as 1 and 0 respectively)
		// the names to show (5 names)
		// the answer array, represented as an array of ints, e.g. [12,12,6,6,0]
	
	var nQuestions;
	var studentsArrays;

	if (arraySizeCondition === 'high'){
		nQuestions = 12;
		// Each participant sees all arrays once!
		// (there's 20 trials and 20 possible arrays)
		studentsArrays = _.shuffle([
			[12, 12, 12, 12, 12],
			[12, 12, 12,  3,  3],
			[12, 12, 12,  9,  9],
			[12, 12, 12,  0,  0],
			[12, 12,  9,  9,  9],
			[12, 12,  9,  3,  3],
			[12, 12,  9,  0,  0],
			[12, 12,  3,  3,  3],
			[12, 12,  3,  0,  0],
			[12, 12,  0,  0,  0],
			[ 9,  9,  9,  9,  9],
			[ 9,  9,  9,  3,  3],
			[ 9,  9,  9,  0,  0],
			[ 9,  9,  3,  3,  3],
			[ 9,  9,  3,  0,  0],
			[ 9,  9,  0,  0,  0],
			[ 3,  3,  3,  3,  3],
			[ 3,  3,  3,  0,  0],
			[ 3,  3,  0,  0,  0],
			[ 0,  0,  0,  0,  0]
		]);
	} else if (arraySizeCondition === 'info'){
			nQuestions = 12;
			studentsArrays = _.shuffle([
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
				[3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
				[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
				[9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0],
				[9, 9, 9, 9, 3, 3, 3, 0, 0, 0, 0],
				[9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3],
				[9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0],
				[9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3],
				[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
				[12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0],
				[12, 12, 12, 12, 3, 3, 3, 0, 0, 0, 0],
				[12, 12, 12, 12, 3, 3, 3, 3, 3, 3, 3],
				[12, 12, 12, 12, 9, 9, 9, 0, 0, 0, 0],
				[12, 12, 12, 12, 9, 9, 9, 3, 3, 3, 3],
				[12, 12, 12, 12, 9, 9, 9, 9, 9, 9, 9],
				[12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0],
				[12, 12, 12, 12, 12, 12, 12, 3, 3, 3, 3],
				[12, 12, 12, 12, 12, 12, 12, 9, 9, 9, 9],
				[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
			]);

	} else if (arraySizeCondition === 'low'){
		nQuestions = 6;
		studentsArrays = _.shuffle([
			[0, 0, 0, 0, 0],
			[1, 1, 0, 0, 0],
			[1, 1, 1, 0, 0],
			[1, 1, 1, 1, 1],
			[5, 5, 0, 0, 0],
			[5, 5, 1, 0, 0],
			[5, 5, 1, 1, 1],
			[5, 5, 5, 0, 0],
			[5, 5, 5, 1, 1],
			[5, 5, 5, 5, 5],
			[6, 6, 0, 0, 0],
			[6, 6, 1, 0, 0],
			[6, 6, 1, 1, 1],
			[6, 6, 5, 0, 0],
			[6, 6, 5, 1, 1],
			[6, 6, 5, 5, 5],
			[6, 6, 6, 0, 0],
			[6, 6, 6, 1, 1],
			[6, 6, 6, 5, 5],
			[6, 6, 6, 6, 6]
		]);
	} 
	}

	// Create an array with 10 zeros and 10 ones and shuffle it.
	// These represent whether the participant is in the high or low
	// condition in each trial. 
	// Total of 20 trials.
	const conditions = _.shuffle(Array(10).fill(1).concat(Array(10).fill(0)));
	var names = [
		"John", "Lisa", "Amy", "Daniel", "Alex", "Tina", "Mia", "Julia", "Tim", "Johann", 
		"Lesly", "Julian", "Chris", "Marie", "Lisanne", "Thomas", "Pablo", "Rebecca", 
		"Theresa", "Susanne", "Jan", "Nico"
	];

	
	// Construct the array with the trial information
	var trials = [];
	for (var i=0; i < conditions.length; i++){
		var trial = {
			// Get a condition from shuffled list of conditions
			condition: conditions[i],
			// Sample 5 student names at random
			names: _.sampleSize(names, studentsArrays.length),
			// Get an answers array from shuffled list
			studentsArray: studentsArrays[i],
			nQuestions: nQuestions,
			// Save the arraySizeCondition
			arraySizeCondition: arraySizeCondition
		};
		trials.push(trial);
	}

	return _.shuffle(trials)
}


// Possible conditions:
// 	'wideShort', (this is the original experiment)
// 	'wideLong',
// 	'narrowShort',
// 	'narrowLong'
const arraySizeCondition = 'narrowLong'

// Called 'questions' cause that is the name of the appropriate
// prop in the CompletionScreen component
const questions = [
	"Describe these results of Riverside so as to make it appear as if there is a <b>low</b> success rate without lying.",
	"Describe these results of Green Valley so as to make it appear as if there is a <b>high</b> success rate without lying."
];
	
const choices = [
	['Some', 'All', 'None', 'Most'],
	['some', 'all', 'none', 'most'],
	['right', 'wrong']
];


export default {
	name: 'App',
	data() {
		return {
			// Shuffle presentation order of trials
			// This is in data and so it is only ran once
			trials: constructTrials(arraySizeCondition),
			// Shuffle order of choices in select box
			// (Only done once per participant)
			choices: _.map(choices, _.shuffle),
			// The two questions
			questions: questions,
			sampleTrial: defineSampleTrial(arraySizeCondition),
			arraySizeCondition: arraySizeCondition
		};
	},
	components: {
		AnswersTable,
		SelectionScreen
	},
	methods: {
		constructTrials,
		defineSampleTrial
	},
	computed: {
		// Expose lodash to template code
		_() {
			return _;
		}
	}
};

</script>
