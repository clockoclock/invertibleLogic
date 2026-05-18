# invertibleLogic
invertible logic verified by python

## Version Revision Record
| Version | Date | Modification | Description | Author |
| :---: | :---: | :--- | :--- | :---: |
| v1.0 | 2026.5.14 | simpleLogicAND.py <br> simpleLogicAND_fixedOutput.py <br> simpleLogicOR.py <br> simpleLogicOR_fixedOutput.py | Upload and organize into the folder *simpleLogic* | WHH |
|  | 2026.5.14 | PuLP_AND.py <br> PuLP_OR.py | Upload and organize into the folder *linearProgramming* | WHH |
|  | 2026.5.14 | PuLP_AND3.py | Upload and organize into the folder *linearProgramming* | WHH |
|  | 2026.5.14 | complexLogicAND3.py <br> complexLogicAND3_fixedOutput0.py <br> complexLogicAND3_fixedOutput1.py | Upload and organize into the folder *complexLogic* | WHH |
|  | 2026.5.14 | complexLogicAND3_auxiliaryConnect.py | Upload and organize into the folder *complexLogic* | WHH |
|  | 2026.5.14 | complexLogicAND3_annealingTrace.py <br> complexLogicAND3_auxiliaryConnect_annealingTraceAnalysis.py | Upload and organize into the folder *annealingTrace* | WHH |
| v1.1 | 2026.5.18 | Note: The original annealing strategy failed to execute and is now uniformly fixed | The involved folders include *simleLogic*, *complexLogic* and *annealingTrace* | WHH |
|  | 2026.5.18 | complexLogicAND3_compactModel.py <bar> complexLogicAND3_compactModel_hamilitonian.py <bar> complexLogicAND3_compactModel_annealingTrace.py <bar> complexLogicAND3_fixedOutput.py | Upload and organize into the folder *complexLogic* and *annealingTrace* | WHH |
|  | 2026.5.18 | complexLogicAND3_auxiliaryConnect_hamilitonian.py <bar> complexLogicAND3_auxiliaryConnect_annealingTrace.py <bar> complexLogicAND3_auxiliaryConnect_fixedOutput.py | Upload and organize 
into the folder *complexLogic* and *annealingTrace* | WHH |

## 2026.5.13
Upload 2 files (simpleLogicAND.py and simpleLogicAND_fixedOutput.py). Here are some details about context.

1. simpleLogicAND.py

    This file utilizes the annealing algorithm to implement the reversible logic of a 2-input AND gate, and adopts 1000 annealing rounds to verify the convergence state of the AND logic.
2. simpleLogicAND_fixedOutput.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 0 or 1.

## 2026.5.13
Upload 2 files (simpleLogicOR.py and simpleLogicOR_fixedOutput.py). Here are some details about context.

3. simpleLogicOR.py

    This file utilizes the annealing algorithm to implement the reversible logic of a 2-input OR gate, and adopts 1000 annealing rounds to verify the convergence state of the OR logic.
4. simpleLogicOR_fixedOutput.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 0 or 1.

## 2026.5.14
Upload 2 files (PuLP_AND.py and PuLP_OR.py). Here are some details about the context.

5. PuLP_AND.py

    This file uses linear programming to solve for the coupling coefficients and bias terms of the AND logic gate.

6. PuLP_OR.py

    This file uses linear programming to solve for the coupling coefficients and bias terms of the OR logic gate.

## 2026.5.14
Upload 4 files (PuLP_AND3.py, complexLogicAND3.py, complexLogicAND3_fixedOutput0.py and complexLogicAND3_fixedOutput1.py). Here are some details about the context

7. PuLP_AND3.py

    This file uses linear programming to solve for the coupling coefficients and bias terms of the AND logic gate.

8. complexLogicAND3.py

    This file utilizes the annealing algorithm to implement the reversible logic of a 3-inputs AND gate, and adopts 10000 annealing rounds to verify the convergence state of the 3-inputs AND logic.

9. complexLogicAND3_fixedOutput0.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 0.

10. complexLogicAND3_fixedOutput1.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 1.

## 2026.5.14
Upload 3 files (complexLogicAND3_auxiliaryConnect.py, complexLogicAND3_annealingTrace.py and complexLogicAND3_auxiliaryConnect_annealingTraceAnalysis.py). Here are some details about the context

11. complexLogicAND3_auxiliaryConnect.py

    This file implements the 3-input AND gate through auxiliary bit connection.

12. complexLogicAND3_annealingTrace.py

    Trace analysis of complexLogicAND3.py.

13. complexLogicAND3_auxiliaryConnect_annealingTraceAnalysis.py

    Trace analysis of complexLogicAND3_auxiliaryConnect.py. Something may be wrong in this file

## 2026.5.18
Upload 4 files (complexLogicAND3_compactModel.py, complexLogicAND3_compactModel_hamilitonian.py, complexLogicAND3_compactModel_annealingTrace.py and complexLogicAND3_compactModel_fixedOutput.py). Here are some details about the context

14. complexLogicAND3_compactModel.py

    This file implements the 3-inputs AND gate through overlap the output of the first AND gate and the input of the second AND gate. And only 5 spins are used in this structure.

15. complexLogicAND3_compactModel_hamilitonian.py

    This file is used to calculate the hamilitionian of proposed structure.

16. complexLogicAND3_compactModel_annealingTrace.py

    Trace analysis of complexLogicAND3_compactModel.py

17. complexLogicAND3_compactModel_fixedOutput.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 0 or 1

## 2026.5.18
Upload 4 files (complexLogicAND3_auxiliaryConnect_hamilitonian.py, complexLogicAND3_auxiliaryConnect_annealingTrace.py and complexLogicAND3_auxiliaryConnect_fixedOutput.py). Here are some details about the context

18. complexLogicAND3_auxiliaryConnect_hamilitonian.py

    This file is used to calculate the hamilitionian of proposed structure.

19. complexLogicAND3_auxiliaryConnect_annealingTrace.py

    Trace analysis of complexLogicAND3_auxiliaryConnect.py

20. complexLogicAND3_auxiliaryConnect_fixedOutput.py

    This file verifies the reversibility of the logic in reverse by fixing the output to 0 or 1
