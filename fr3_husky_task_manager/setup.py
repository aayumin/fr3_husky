from setuptools import find_packages, setup

package_name = 'fr3_husky_task_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuminlim',
    maintainer_email='ckrgksakdmac@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'apple_vision_pro = fr3_husky_task_manager.apple_vision_pro:main',
            'keyboard_move = fr3_husky_task_manager.keyboard_move:main',
            'husky_pedal = fr3_husky_task_manager.husky_pedal:main',
            'move_to_joint = fr3_husky_task_manager.move_to_joint:main',
            'task_space_move = fr3_husky_task_manager.task_space_move:main',
            'task_space_delta_move = fr3_husky_task_manager.task_space_delta_move:main',
            'contact_guarded_motion = fr3_husky_task_manager.contact_guarded_motion:main',
            'nut_tightening = fr3_husky_task_manager.nut_tightening:main',
            'gripper_move = fr3_husky_task_manager.gripper_move:main',
            'log = fr3_husky_task_manager.live_log:main',
            'image = fr3_husky_task_manager.image_streaming:main',
        ],
    },
)
