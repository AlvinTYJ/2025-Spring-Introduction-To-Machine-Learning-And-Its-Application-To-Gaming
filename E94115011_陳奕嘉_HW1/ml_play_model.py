import random
import os.path
import pickle

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.ai_name = ai_name
        filename = 'DZKmodel.pickle'
        filepath = os.path.join(os.path.dirname(__file__), filename)
        self.model = pickle.load(open(filepath, 'rb'))
        
        self.ball_served = False
        self.previous = (100, 395)
        self.move_counter = 0
        self.direction = random.choice(["MOVE_LEFT", "MOVE_RIGHT"])
        self.rnd2 = random.randint(0, 7)

    def update(self, scene_info, *args, **kwargs):
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not scene_info["ball_served"]:
            if self.move_counter < self.rnd2:
                if self.direction == "MOVE_LEFT":
                    command = "MOVE_LEFT"
                elif self.direction == "MOVE_RIGHT":
                    command = "MOVE_RIGHT"
                
                self.move_counter += 1
                return command
            
            else:
                rnd_serve = random.random()
                if rnd_serve < 0.5:
                    command = "SERVE_TO_LEFT"
                else:
                    command = "SERVE_TO_RIGHT"
                
                self.ball_served = True
            
        velocity_x = scene_info['ball'][0] - self.previous[0]
        velocity_y = scene_info['ball'][1] - self.previous[1]
        command = self.model.predict([[scene_info["ball"][0], scene_info["ball"][1], scene_info["platform"][0], velocity_x, velocity_y]])
        self.previous = scene_info['ball']
            
        if command == 0: return "NONE"
        elif command == -1: return "MOVE_LEFT"
        elif command == 1: return "MOVE_RIGHT"

    def reset(self):
        self.ball_served = False
        self.move_counter = 0
        self.platform_direction = random.choice(["MOVE_LEFT", "MOVE_RIGHT"])
        self.move_limit = random.randint(0, 7)
        self.previous_ball_position = (100, 395)