<?php

$state=$_POST["state"];

$xx=exec("sudo python /object_tracking/master.py $state");

?>