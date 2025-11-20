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
	<!-- Actual experimental trials -->
	<template v-for="(trial, i) of trials">

		<ForcedChoiceScreen
			:key='i'
			:qud="questions[0]"
			:question="`${trial.Q1} of the students got ${trial.Q2} of the answers ${trial.A}`"
			:options="trial.options"
			:progress="i / trials.length"
		>

			<!-- Table + data recorder -->
			<template #stimulus>
					<AnswersTable :trialData="trial" />
					<Record
						:data="trial"
					/>
					<!-- 🚀 DEBUG PRINT -->
					<pre style="background:#eee; padding:5px; font-size:12px;">
						trial {{ i }}:
						studentsArray = {{ trial.studentsArray }}
						names: {{ trial.names }}
						condition: {{ trial.condition }}
					</pre>
			</template>

		</ForcedChoiceScreen>

	</template>

	<PostTestScreen />
	<SubmitResultsScreen />
  </Experiment>
</template>

<script>
import _ from 'lodash';
import AnswersTable from '@/Answers.vue';
import SelectionScreen from '@/SelectionScreen.vue';
import itemsRaw from '../items/final_listener_items.csv';

// --- helpers ---
// If observation in CSV is already an array, you can skip this and just use itemsRaw directly.
function normalizeItems(rawItems) {
  return rawItems.map((row, idx) => {
    let obs = row.observation;

    // If observation is a string (e.g., "12,12,0,0,0"), parse it into an array of numbers.
    if (typeof obs === "string") {
      obs = obs
        .replace(/[\[\]\(\)]/g, "")
        .split(/[,\s]+/)
        .filter((x) => x !== "")
        .map((x) => Number(x));
    }

    return {
      id: row.id != null ? row.id : idx,
      Q1: row.Q1,
      Q2: row.Q2,
      A: row.A,
      condition: row.condition,
      observation: obs,
    };
  });
}

const items = normalizeItems(itemsRaw);

function defineSampleTrial(item) {
  // item is one row from itemsTable: { Q1, Q2, A, observation, condition, ... }

  const namesPool = [
    "John", "Lisa", "Amy", "Daniel", "Alex", "Tina", "Mia", "Julia", "Tim", "Johann",
    "Lesly", "Julian", "Chris", "Marie", "Lisanne", "Thomas", "Pablo", "Rebecca",
    "Theresa", "Susanne", "Jan", "Nico"
  ];

  const nStudents = item.observation.length;

  // Build a sample trial object matching the shape used in constructTrials
  const trial = {
    condition: item.condition,
    names: _.sampleSize(namesPool, nStudents),
    studentsArray: item.observation,
    nQuestions: 12,        // or infer from observation if you prefer
    Q1: item.Q1,
    Q2: item.Q2,
    A: item.A
  };

  return trial;
}

function constructTrials(itemsTable) {
  // Pool of possible student names
	var namesPool = [
    "John", "Lisa", "Amy", "Daniel", "Alex", "Tina", "Mia", "Julia", "Tim", "Johann",
    "Lesly", "Julian", "Chris", "Marie", "Lisanne", "Thomas", "Pablo", "Rebecca",
    "Theresa", "Susanne", "Jan", "Nico"
  ];
  	const choiceOptions = ["Student", "Teacher", "Principal", "I don't know"];

  // Turn each row in the items table into a trial object
  	var trials = itemsTable.map(function(item) {
    var obs = item.observation;
    var nStudents = obs.length;

    // If you always have 12 questions, fix nQuestions = 12.
    // Otherwise you can infer it from the max count:
    // var nQuestions = Math.max.apply(null, obs);
    var nQuestions = 12;

    return {
      // Ground-truth speaker type for this item
      condition: item.condition,          // "high", "low", or "info"

      // Names: sample as many names as there are students in the observation
      names: _.sampleSize(namesPool, nStudents),

      // The counts per student (e.g., [12, 12, 0, 0, 0])
      studentsArray: obs,

      // Total number of questions in the exam
      nQuestions: nQuestions,

      // Keep linguistic info for sentence construction
      Q1: item.Q1,   // first quantifier
      Q2: item.Q2,   // second quantifier
      A:  item.A,     // "right" / "wrong"
	  options: _.shuffle(choiceOptions)
    };
  });

  // Randomize trial order before returning
  return _.shuffle(trials);
}


// Possible conditions:
// 	'wideShort', (this is the original experiment)
// 	'wideLong',
// 	'narrowShort',
// 	'narrowLong'
const arraySizeCondition = 'narrowLong'

// Called 'questions' cause that is the name of the appropriate
// prop in the CompletionScreen component
// Question shown above the report
const questions = [
  "Who is most likely to have said this report?"
];

// Possible response options (will be shuffled once per participant)
const choiceOptions = ["Student", "Teacher", "Principal", "I don't know"];


export default {
  name: "App",
  components: {
    AnswersTable,
    SelectionScreen,
  },
  data() {
    const trials = constructTrials(items);

    return {
      trials,
      choices: [_.shuffle(choiceOptions)],
      questions,
      sampleTrial: defineSampleTrial(items[0]),
      items,
    };
  },
  methods: {
    constructTrials,
    defineSampleTrial,
  },
  computed: {
    _() {
      return _;
    },
  },
};

</script>
