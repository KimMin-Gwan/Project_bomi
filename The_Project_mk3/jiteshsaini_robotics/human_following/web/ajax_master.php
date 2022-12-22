<?php

$state=$_POST["state"];

$xx=exec("sudo python /human_tracking/master.py $state");

?>