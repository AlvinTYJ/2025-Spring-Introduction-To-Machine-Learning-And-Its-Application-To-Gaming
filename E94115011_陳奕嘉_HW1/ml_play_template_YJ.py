import random
import pickle
import os

def save_game_data(game_data):
    game_info = []
    game_command = []

    for frame in game_data:
        scene_info = {
            'ball': frame['scene_info']['ball'],
            'platform': [frame['scene_info']['platform'][0]]
        }
        game_info.append(scene_info)
        game_command.append(frame['command'])

    data_to_save = {
        'ml': {
            'scene_info': game_info,
            'command': game_command
        }
    }

    filepath = "log/"
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    filename = "savefile.pickle"
    with open(os.path.join(filepath, filename), "wb") as f:
        pickle.dump(data_to_save, f)

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        print(ai_name)
        self.game_data = []
        self.previous_ball = (100, 395)
        self.ball_served = False
        
        self.move_counter = 0
        self.direction = random.choice(["MOVE_LEFT", "MOVE_RIGHT"])

    def update(self, scene_info, *args, **kwargs):
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not scene_info["ball_served"]:
            rnd2 = random.randint(0, 10)
            if self.move_counter < rnd2:
                if self.direction == "MOVE_LEFT":
                    command = "MOVE_LEFT"
                elif self.direction == "MOVE_RIGHT":
                    command = "MOVE_RIGHT"
                
                self.move_counter += 1
                return command
            
            rnd_serve = random.random()
            if rnd_serve < 0.5:
                command = "SERVE_TO_LEFT"
            else:
                command = "SERVE_TO_RIGHT"
                
            self.ball_served = True
            
        else:
            x_now = scene_info["ball"][0]
            y_now = scene_info["ball"][1]
            
            x_prev = self.previous_ball[0]
            y_prev = self.previous_ball[1]
            
            if x_prev == x_now:
                command = "NONE"
            else:
                m = (y_prev - y_now) / (x_prev - x_now)
            
                if y_now > y_prev:
                    x_predict = (400 - y_now) / m + x_now
                
                    if x_predict > 200:
                        x_predict = 200 - (x_predict - 200)
                    if x_predict < 0:
                        x_predict = abs(x_predict)
                else:
                    x_predict = 100
                
                rnd = random.randint(18, 22)
                if x_predict < (scene_info["platform"][0] + rnd):
                    command = "MOVE_LEFT"
                elif x_predict > (scene_info["platform"][0] + rnd):
                    command = "MOVE_RIGHT"
                else:
                    command = "NONE"
                        
        self.previous_ball = scene_info["ball"]

        self.game_data.append({
            'scene_info': scene_info,
            'command': command
        })

        return command

    def reset(self):
        save_game_data(self.game_data)

        self.ball_served = False
        self.prev_ball = [100, 395]
        self.game_data = []