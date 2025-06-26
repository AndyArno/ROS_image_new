# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.0.2)

# 设置项目名称，必须与 package.xml 中的 <name> 一致
project(image_new)

# 查找catkin和本项目需要的其他ROS包
find_package(catkin REQUIRED COMPONENTS
  rospy
)

# 声明catkin包的属性，供其他包使用
# CATKIN_DEPENDS 指明了依赖 rospy
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES image_new
  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

###########
## 安装规则 ##
###########
# 这部分内容是新增的，用于正确安装项目文件

# 安装Python可执行脚本
# 让ROS系统能找到并执行你的Python节点
catkin_install_python(PROGRAMS
  src/recognize.py
  src/transformer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch文件夹及其中的所有文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装config文件夹及其中的所有文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)