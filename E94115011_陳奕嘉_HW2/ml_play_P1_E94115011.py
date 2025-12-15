import os.path
import pickle

class MLPlay:
	def __init__(self, ai_name, *args, **kwargs):
		filename = 'model_P1_E94115011.pickle'
		filepath = os.path.join(os.path.dirname(__file__), filename)
		self.model = pickle.load(open(filepath, "rb"))
		self.ball_served = False
		self.side = ai_name
		
	def update(self, scene_info, *args, **kwargs):
		if scene_info["status"] != "GAME_ALIVE":
			return "RESET"

		if not self.ball_served:
			self.ball_served = True
			command = "SERVE_TO_LEFT"
		else:
			ball_x = scene_info["ball"][0]
			ball_y = scene_info["ball"][1]
			velocity_x = scene_info["ball_speed"][0]
			velocity_y = scene_info["ball_speed"][1]
			platform_x = scene_info["platform_1P"][0]
			num = self.model.predict([[ball_x, ball_y, velocity_x, velocity_y, platform_x]])
   
			if num  == -1: command = "MOVE_LEFT"
			elif num == 1: command = "MOVE_RIGHT"
			else: command = "NONE"
		return command

	def reset(self):
		self.ball_served = False
