from dummy_classes import ExoBoot, OccipitalLobe

class HolisticExoSkeleton: # holistic as in it also includes vision now!
    def __init__(self, LeftBoot : ExoBoot, RightBoot : ExoBoot, VisionSystem : OccipitalLobe):
        self.left_boot = LeftBoot
        self.right_boot = RightBoot
        self.vision_system = VisionSystem
    def read_sensors(self):
        self.left_boot.read_sensors()
        self.right_boot.read_sensors()
        self.vision_system.sense()
    def run_level_ground_profile(self, external_read = True):
        self.left_boot.run_collins_profile(external_read=external_read)
        self.right_boot.run_collins_profile(external_read=external_read)
    def run_stair_ascent_profile(self, heel: bool): #NOTABLY ONLY ON THE FIRST STEP FOR NOW, AND ONLY ON THE RIGHT FOOT
        if heel:
            self.right_boot.run_stair_profile_heel()
        else:
            self.right_boot.run_stair_profile_toe()

    def run_logic(self):
        visual_state = self.vision_system.state # tells you if level ground, toe, or heel
        if visual_state.level_ground:
            self.run_level_ground_profile()
        elif visual_state.heel:
            self.run_stair_ascent_profile(heel=True)
        elif visual_state.toe:
            self.run_stair_ascent_profile(heel=False)

