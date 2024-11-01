
# 目标
- 使用大模型来生成行为树，用于控制角色的行为

## 1 目标分解
- 1.1 将问题范围缩小，文本描述的是虚拟环境中小车驾驶的行为，如:车辆正常保持车道行驶，前方遇到车辆，如果可以就变道超车,否则正常保持车道行驶
- 1.2 以上问题转换成行为树的关键是找出语义中蕴含的条件，也就是 前方是否有车辆 
- 1.3 以Json形式表示
```

// 需要注意的是
{"ROOT":{
    "TYPE": "SELECTOR",
    "NAME": "xxx",
    "CHILDREN":[
        {
            "TYPE": "SEQUENCE",
            "NAME": "",
            "CHILDREN": [
                {
                    "TYPE": "BEHAVIOR",
                    "FUNCTION_CALL": {
                        "DESCRIPTION": "判断前方是否有车辆&满足超车条件"
                    }
                },
                {
                    "TYPE": "BEHAVIOR",
                    "FUNCTION_CALL": {
                        "DESCRIPTION": "进行超车行为"
                    }
                }
                
            ]
        },
        {
            "TYPE": "BEHAVIOR",
                "FUNCTION_CALL": {
                    "DESCRIPTION": "保持车道行驶"
                }
        }
    ]
}}
```
