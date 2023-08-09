           
% Initialize Observation settings
ObservationInfo = rlFiniteSetSpec([1:100]);
ObservationInfo.Name = 'position';
ObservationInfo.Description = 'x-axis position of the particle';

%initialize the action space.
% note we put 20 gaussian at one time. each gaussian has position and width.
% we put upper/lower bound to the position and width. [1, N] and [0.5, 2]
lowerLimits(1,1:20)=0.5;
lowerLimits(2,1:20)=0.5;
upperLimits(1,1:20)=90;
upperLimits(2,1:20)=5;

ActionInfo = rlNumericSpec([2,20],'LowerLimit', lowerLimits, 'UpperLimit',upperLimits);
%ActionInfo.Name = 'gaussian position and width';
type myResetFunction.m
type myStepFunction.m
MFPT_env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');

%validateEnvironment(MFPT_env);
