"""
加载地图与视角
"""

import carla

# 连接到CARLA服务器
client = carla.Client("localhost", 2000)
client.set_timeout(100.0)  # 设置超时时间

# 获取可用的地图列表（可选步骤）
print(client.get_available_maps())

# 加载新的地图
client.load_world("Town05")

world = client.get_world()
transform = carla.Transform()

# 设置观察者
spectator = world.get_spectator()
bv_transform = carla.Transform(
    transform.location + carla.Location(z=250, x=0),
    carla.Rotation(yaw=0, pitch=-90),
)
spectator.set_transform(bv_transform)
