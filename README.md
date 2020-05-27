**Training_Sequential.ipynb** contains the latest code for Sequential training.


**Installation**  
    
```
pip3 install torch
pip3 install gym
pip3 install gym[atari]
```
  
    
      
      
**Frame Dimensions** 
```
env.observation_space
```   
Box(210, 160, 3)  
    
      
        
        
**Actions** 
```
 env.unwrapped.get_action_meanings()
 ```
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


**Sample Run 1**
```
python3 pong.py 1
```
```
Score obtained: 0.0
```

**Sample Run 2**
```
python3 pong.py 0

```
```
Enter Number of games
10
```
```
Score        0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	
Frequency    10	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
```
