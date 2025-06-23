import numpy as np
import re
import sys

sys.path.append("../")
from core.fewshot_agent import FewShotAgent # noqa: E402
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class LLMProposer(FewShotAgent):
    def __init__(self,
                 ckt_name_description,
                 params_list,
                 task_context,
                 backend,
                 example_keys,
                 n_proposal=1,
                 ranges=None,
                 max_request_attempt=3,
                 ):
        super().__init__(
            example_keys=example_keys
        )
        self.ckt_name_description = ckt_name_description
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.backend_history = backend
        self.max_request_attempt = max_request_attempt  # TODO: change request to adapt to larger number of proposal;

        self.ranges = ranges
        self.n_proposal = n_proposal

        self.debug_mode = False

        self.num_calls = 0

    def propose_params(self, data_collected):
        if self.n_proposal == 0:
            return []
        else:
            examples = self.generate_examples(data_collected)

            # Propose candidate points within maximum attempts;
            #print(self.max_request_attempt, self.n_proposal)
            params_proposed = []
            for _ in range(self.max_request_attempt):
                #print('EXAMPLES: ', examples)
                prompt = self.generate_prompt(examples, "")
                responses = self.backend.request(prompt)
                params_parsed = self.parse_llm_responses(responses)

                params_proposed += params_parsed

                if self.debug_mode:
                    print("***Prompt***\n")
                    #print(f"{prompt}")

                    print("***Response***\n")
                    #print(f"{responses[0]}")

                if len(params_proposed) >= self.n_proposal:
                    break

            if len(params_proposed) < self.n_proposal:
                raise ValueError("LLM failed to propose enough valid data!!")
            else:
                params_proposed = params_proposed[0:self.n_proposal]
                self.num_calls += 1
                return params_proposed


    def generate_prefix(self, prompt_input):
        # TODO: looks like the GPT seems to be lazy and did not think to much for the given task context and sequence of the prompt_input;
        
        prefix = """ """

        # add more circuits when needed;
        related_ckts = ['Two_Stage_Differential_Amplifier', 'Hysteresis_Comparator']
        
        # Read from NATURAL LANGUAGE txt file
        multiline_history = """ """
        for ckt in related_ckts:
            with open(f'/history_summary/critic_{ckt}.txt', 'r') as file:
                multiline_history = file.read()

        # Comment if running first circuit or to independently optimize each circuit
        if (self.num_calls > 1):
            prefix += ""
            prefix += multiline_history

        prefix += self.task_context


        prefix += "**Examples**\n"

        return prefix

    def generate_suffix(self, prompt_input):
        suffix = """Based on the netlist and demonstrations and the design rules of the two stage differential amplifier circuit, contemplate on how you can adjust the sizing of parameters to achieve the desired specifications.
        Specifically, use common circuit sub-structures between the two structures and then use the sizing rules mentioned for the matching sub-scircuit in the two stage differential amplifier to size different sub-circuits of the folded cascode amplifier.
        Then propose a different set of parameters (all in the range [0,1]) in the format of params in **Examples**. You must not add any comments beyond the recommendation.
        """

        suffix = """ """

        suffix += f"Note that your suggested design should be neither too far from the BEST parameter settings (examples), nor too close to the BEST parameter. Your response should only include the parameters which should be in the format of params in **Examples** and should strictly be between 0 and 1 only and with a precision of 2 atleast!. Your answer should be in the format of params in **Examples** \n "
        return suffix
    
    
    def parse_llm_responses(self, responses):
        # TODO: range of the params is not checked currently;
        # TODO: add support to log the LLM reasoning part;
        params_units = self.params_units
        params_list = self.params_list

        # Extract formatted sub-strings with the format: param=value_param with regular expression;
        # value_param needs to be float number (optionally) followed by one of the units;
        def extract_params(response):
            found_values = {}
            # Create a regular expression pattern for the units, allowing only specified units;
            # Escape units to avoid regex issues;
            units_pattern = f"[{''.join(re.escape(unit) for unit in params_units)}]?"

            for param in params_list:
                # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces;
                match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?)\b", response)
                if match:
                    # Store the found value using the parameter as the key;
                    found_values[param] = match.group(1)
                else:
                    # If any parameter is not found or its value is invalid, return False;
                    return False

            # formatted_output = "".join(f"{param}={value}" for param, value in found_values.items())
            formatted_output = found_values

            return formatted_output
        

        # Extract for every collected response;
        params_parsed = []
        # Response is None means api call has failed in request attempts (default to be 3);
        if responses is None:
            print(f"Api call fails!")
        else:
            for response in responses:
                response_parsed = extract_params(response)
                if response_parsed:
                    params_parsed.append(response_parsed)

        return params_parsed


