<?php

$state=$_POST["state"];

$xx=exec("sudo python The_Project/jiteshsaini_robotics/object_detection/object_detection/master.py $state");

?>
