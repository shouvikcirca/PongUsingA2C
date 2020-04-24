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
