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
			<b>Researcher collecting data</b>: Hening Wang
			<br /><br />
			<b>What is this document?</b>  This document explains what kind of study we’re doing, what your rights are, and what will be done with your data. You should print this page for your records.
			<br /><br />
			<b>Nature of the study.</b> You are invited to participate in a study which involves looking at sets of (fictitious) exam results. You will read descriptions about these results and decide who wrote them. Once you finish, we will have also some brief questions concerning your age, gender, and language background. Your session should last for up to 10 minutes. You will be given full instructions shortly.
			<br /><br />
			<b>Compensation.</b> You will be paid £1.5 for your participation in this study.
			<br /><br />
			<b>Risks and benefits.</b> There are no known risks to participation in this study. Other than the payment mentioned, there are no tangible benefits to you, however you will be contributing to our knowledge about language.  
			<br /><br />
			<b>Confidentiality and use of data.</b> All the information we collect during the course of the research will be processed in accordance with Data Protection Law. In order to safeguard your privacy, we will never share personal information (like your name) with anyone outside the research team. Your data will be referred to by a unique participant number. Please note that we will temporarily collect your participant ID to prevent repeated participation, however we will never share this information with anyone outside the research team. We will store any personal data (i.e., participant ID) securely using the University of Tübingen’s storage systems. The anonymised data collected during this study will be used for research purposes and may be shared with other researchers or with the general public (e.g., we may make it available through the world wide web, or use it in TV or radio broadcasts).
			<br /><br />
			<b>Voluntary participation and right to withdraw.</b> Your participation is voluntary, and you may withdraw from the study at any time and for any reason. If you withdraw from the study after data gathering, we will delete your Prolific ID from the data and there is no penalty or loss of benefits to which you are otherwise entitled. To withdraw from the study after data gathering please contact the research team by providing your prolific participant ID.  
			<br /><br />
			If you have any questions about what you’ve just read, please feel free to ask, or contact us later. You can contact us by email at hening.wang@uni-tuebingen.de.
			<br /><br />
			By accepting this task, you consent to the following:
			<br /><br />
			<span style="margin-left:2em">1. <b>I agree to participate in this study.</b></span>
			<br /><br />
			<span style="padding-left:2em">2. I confirm that I have read and understood <b>how my data will be stored and used.</b></span>
			<br /><br />
			<span style="margin-left:2em">3. I understand that I have the <b>right to terminate this session</b> at any point. If I choose to <b>withdraw after completing the study</b> by contacting the research team until <b>31/03/2026</b>, my <b>Prolific ID will be deleted from the data</b> at that time.</span>
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
			Many events in life can be described in one way or another. Think of describing the same glass as either half full or half empty. In this experiment, we are interested in how <i>you</i> would interpret the intended meaning behind an utterance.
			<br/><br/>
			You will see pictures showing the true results of an exam, along with short reports describing those results. Your task is to decide <b>who</b> made each report, given the true outcomes and the different goals of the possible speakers.
			<br/><br/>
			The next screen will give you more concrete instructions for the upcoming task.
			<button @click="$magpie.nextSlide()">Next</button>
		</Slide>
	
		<Slide>
		<strong>Background:</strong> Recently, Green Valley High School has held several exams. In some exams, students had a high probability of success; in others, the chances of success were lower.
		<br/><br/>
		There are three people who may report about these exams, each with slightly different goals:
		<br/><br/>
		<b>The student</b> wants the exam to sound difficult&mdash;that is, as if students had a low chance of success&mdash;because it would make her own performance seem better in comparison.
		<br/><br/>
		<b>The teacher</b> wants the exam to sound easy&mdash;that is, as if students had a high chance of success&mdash;because it would reflect positively on the quality of teaching and the students’ overall preparedness.
		<br/><br/>
		<b>The principal</b> provides an objective report that aims to describe the exam as it was, without making it sound particularly hard or easy.
		<br/><br/>
		In the following practice trails, you will see examples of exam results. The tables are showing the exam results of students. 
			A "<i style="color:#13AC38">&#10004;</i>" indicates that a task was answered correctly, 
			and a "<i style="color:#B12810">&#10008;</i>" that it was answered incorrectly.You will first try to make your own report as if you were one of the three speakers, and then you will see example reports made by each of the three speakers.
		<button @click="$magpie.nextScreen()">Next</button>
		</Slide>
	</Screen>
	<template v-for="(sampleTrial, i) in sampleTrials">
		<SelectionScreen
			:question="speaker_questions[sampleTrial.condition]"
			text="%s of the students got %s of the answers %s"
			:options="speaker_choices"
			:key='i'
			:progress='i / sampleTrials.length'
		>

			<template #stimulus>
				<AnswersTable :trialData='sampleTrial' />
				<Record :data="sampleTrial" />
			</template>
		</SelectionScreen>
	</template>
	<Screen>
		<Slide>
			Great! You've completed the training phase as a spekaer. We will move on to the main part of the experiment next, where you will take the role of a listener.
			<br/><br/>
			Remember: the tables are showing the exam results of students. 
			A "<i style="color:#13AC38">&#10004;</i>" indicates that a task was answered correctly, 
			and a "<i style="color:#B12810">&#10008;</i>" that it was answered incorrectly.
			<br/><br/>
			From the table, note that each person reports with a different goal:
			<br/><br/>
				The <b>student</b> wants the exam to sound <b>difficult</b>.
			<br/><br/>
				The <b>teacher</b> wants the exam to sound <b>easy</b>.
			<br/><br/>
				The <b>principal</b> provides an <b>objective</b> report.
			<br/>
			Your task is to decide <b>who</b> said the report.
			<br/><br/>
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
					 <!--
					<pre style="background:#eee; padding:5px; font-size:12px;">
						trial {{ i }}:
						studentsArray = {{ trial.studentsArray }}
						names: {{ trial.names }}
						condition: {{ trial.condition }}
					</pre>
					-->
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
	const Q1 =
      typeof row.Q1 === "string" && row.Q1.length > 0
        ? row.Q1.charAt(0).toUpperCase() + row.Q1.slice(1)
        : row.Q1;

    return {
      id: row.id != null ? row.id : idx,
      Q1: Q1,
      Q2: row.Q2,
      A: row.A,
      condition: row.condition,
      observation: obs,
    };
  });
}

const items = normalizeItems(itemsRaw);

function defineSampleSpeakerTrial(arraySizeConditions) {
  // arraySizeConditions is an array like ['high', 'info', 'low']
  const trials = arraySizeConditions.map(function (arraySizeCondition) {
    if (arraySizeCondition === "high") {
      return {
        condition: "sample.high",
        names: ["Marie", "Nico", "Mia", "John", "Alex"],
        studentsArray: [12, 3, 0, 0, 0],
        nQuestions: 12,
      };
    } else if (arraySizeCondition === "info") {
      return {
        condition: "sample.info",
        names: ["Julian", "Chris", "Marie", "Lisanne", "John"],
        studentsArray: [9, 9, 9, 9, 9],
        nQuestions: 12,
      };
    } else if (arraySizeCondition === "low") {
      return {
        condition: "sample.low",
        names: ["Marie", "Nico", "Mia", "John", "Alex"],
        studentsArray: [12, 12, 9, 9, 9],
        nQuestions: 12,
      };
    }
    // If some unknown condition sneaks in, return null to filter later
    return null;
  });

  // Filter out any null entries (in case of unexpected conditions)
  const validTrials = trials.filter(function (t) {
    return t !== null;
  });

  console.log("Sample speaker trials:", validTrials);
  return validTrials;
}


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
  	const choiceOptions = ["Student", "Principal", "Teacher"];

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
	  options: choiceOptions
    };
  });

  // Randomize trial order before returning
  return _.shuffle(trials);
}

// Called 'questions' cause that is the name of the appropriate
// prop in the CompletionScreen component
// Question shown above the report
const questions = [
  "Who is most likely to have said this report?"
];

// Possible response options (will be shuffled once per participant)
const choiceOptions = ["Student", "Teacher", "Principal"];

const speaker_questions = {
	"sample.low": "Imagine you were a <b>student</b>, describe these results of this exam so as to make it appear as if there is a <b>low</b> success rate without lying.",
	"sample.info": "Imagine you were a <b>principal</b>, describe these results of this exam as <b>accurately</b> as possible without making it sound particularly easy or difficult.",
	"sample.high": "Imagine you were a <b>teacher</b>, describe these results of this exam so as to make it appear as if there is a <b>high</b> success rate without lying."
};
	
const speaker_choices = [
	['Some', 'All', 'None', 'Most'],
	['some', 'all', 'none', 'most'],
	['right', 'wrong']
];

const sampleTrial_conditions = ['high', 'info', 'low'];

export default {
  name: "App",
  components: {
    AnswersTable,
    SelectionScreen,
  },
  data() {
    const trials = constructTrials(items);
	const sampleTrials = defineSampleSpeakerTrial(sampleTrial_conditions);
    return {
      trials,
      choices: [_.shuffle(choiceOptions)],
	  speaker_choices: _.map(speaker_choices, _.shuffle),
      questions,
	  speaker_questions,
      sampleTrials,
      items,
    };
  },
  methods: {
    constructTrials,
    defineSampleSpeakerTrial,
	defineSampleTrial,
  },
  computed: {
    _() {
      return _;
    },
  },
};

</script>
