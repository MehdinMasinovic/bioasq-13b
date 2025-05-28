# Write shell script to run evaluation measures
# It should have the following structure:
# Take two filenames, the first is the ground truth file (A) and the second is the predicted file (B)
# The script should pass these files to the eval_helpers.py script which then writes to evaluation_measures/results/mesh/A.txt and evaluation_measures/results/mesh/B.txt
# The script should then run the following java command to map the mesh files:
# java -Xmx10G -cp ../Evaluation-Measures/flat/BioASQEvaluation/dist//BioASQEvaluation.jar converters.MapMeshResults evaluation_measures/mesh/mapping.txt evaluation_measures/results/mesh/A.txt evaluation_measures/results/mapped/B.txt
# And finally, it should run the following command to evaluate the results:
# java -Xmx10G -cp ../Evaluation-Measures/flat/BioASQEvaluation/dist//BioASQEvaluation.jar evaluation.Evaluator evaluation_measures/results/mapped/A.txt evaluation_measures/results/mapped/B.txt
# The above prints the results to the console which should be redirected to evaluation_measures/results/output/A_results.txt. 

#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ground_truth_file> <predicted_file> <output_name>"
    exit 1
fi
GROUND_TRUTH_FILE=$1
PREDICTED_FILE=$2
OUTPUT_NAME=$3
EVAL_HELPERS_SCRIPT="evaluation_measures/eval_helpers.py"
MESH_DIR="evaluation_measures/results/mesh"
MAPPED_DIR="evaluation_measures/results/mapped"
OUTPUT_DIR="evaluation_measures/results/output"
python3 $EVAL_HELPERS_SCRIPT $GROUND_TRUTH_FILE $PREDICTED_FILE $OUTPUT_NAME # venv/bin/python
mkdir -p $MESH_DIR
mkdir -p $MAPPED_DIR
mkdir -p $OUTPUT_DIR
java -Xmx10G -cp ../Evaluation-Measures/flat/BioASQEvaluation/dist//BioASQEvaluation.jar converters.MapMeshResults evaluation_measures/mesh/mapping.txt $MESH_DIR/${OUTPUT_NAME}_pred.txt $MAPPED_DIR/${OUTPUT_NAME}_pred.txt
java -Xmx10G -cp ../Evaluation-Measures/flat/BioASQEvaluation/dist//BioASQEvaluation.jar converters.MapMeshResults evaluation_measures/mesh/mapping.txt $MESH_DIR/${OUTPUT_NAME}_gold.txt $MAPPED_DIR/${OUTPUT_NAME}_gold.txt
java -Xmx10G -cp ../Evaluation-Measures/flat/BioASQEvaluation/dist//BioASQEvaluation.jar evaluation.Evaluator $MAPPED_DIR/${OUTPUT_NAME}_gold.txt $MAPPED_DIR/${OUTPUT_NAME}_pred.txt > $OUTPUT_DIR/${OUTPUT_NAME}_results.txt
echo "Evaluation completed. Results are saved in $OUTPUT_DIR/${OUTPUT_NAME}_results.txt"
# End of script
# Make sure to give execute permission to the script before running it:
# chmod +x run_eval_measures.sh
# You can run the script like this:
# ./run_eval_measures.sh ground_truth.txt predicted.txt