# from src.workflows.multimodelauth import MultiModalVerificationWorkflow
#
# workflow = MultiModalVerificationWorkflow()
#
# user_id = input("Enter User ID: ")
# result = workflow.verify(user_id)
#
# print("\nFINAL RESULT")
# print(result)

from src.workflows.multimodelauth import MultiModalVerificationWorkflow
import sys

workflow = MultiModalVerificationWorkflow()

if len(sys.argv) > 1:
    user_id = sys.argv[1]
else:
    user_id = input("Enter User ID: ")

result = workflow.verify(user_id)

print("\nFINAL RESULT")
print(result)